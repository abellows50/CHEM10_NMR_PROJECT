from vispy import scene
from vispy.visuals import transforms
from vispy import app
from vispy import io

import time 

import numpy as np

import signal
import sys

import threading ## Give the user control over the simulation!
shutdown = threading.Event()
save_next_frame = threading.Event()

name = ""
name_mutex = threading.Lock()

console_mutex = threading.Lock()

run_sim = threading.Event()

## Handle exits gracefully instead of throwing a system_crashed error on ^C
def handler(sig, frame):
    shutdown.set()

# register for Ctrl+C (SIGINT)
signal.signal(signal.SIGINT, handler)


### Helper Functions ###
def magnitude(vector):
    return np.sqrt(np.dot(vector,vector))

def cprint(*args, **kwargs):
    with console_mutex:
        print(*args, **kwargs)

class Sample:
    T2 = 0.5 # Transverse Relaxation Time Constant
    T1 = 1 # Longitudinal Relaxation Time Constant
    R1 = 1/T1 # Relaxtation constant
    R2 = 1/T2 # Relaxtation constant
    gamma = 10 # Gyromagnetic ratio


class Bloch_Simulator:
    # pulse variables
    class Pulse_info:
        pulse_duration = 0
        pulse_start = -1
        pulse_onging = 0

    def __init__(self):
        self.sample = Sample()
        self.pulse = self.Pulse_info()
        self.pulse_mutex = threading.Lock()


        self.B_device = np.array([1.,1.,5.]) # Magnetic Field Vector of the device (points in the positive z direction)
        self.B_eff = self.B_device.copy()
        self.B_eff_mutex = threading.Lock()

        self.M = self.B_device/magnitude(self.B_device) # Net Magnitization Vector
        self.M0 = magnitude(self.M)


        self.dt = 0.001 # change in time per step

        return
    
    def set_renderer(self, renderer):
        self.renderer = renderer

    ## This function takes the current net magnetization, starting magnetization, B field, both decay constants and the gyromagnetic ratio
    ## It returns the differential of M computed via the Bloch equations
    ## This function appears to be highly sensitive to input for certain M,B,R,gamma values. I believe this is a sensitive dependence on
    ## initial conditions problem
    def calculate_bloch_differentials(self) -> np.array:
        with self.B_eff_mutex: ## entering critical section -- B must be invariant
            dmxdt = self.sample.gamma*np.cross(self.M,self.B_eff)[0] - self.R2*self.M[0]
            dmydt = self.sample.gamma*np.cross(self.M,self.B)[1] - self.R2*self.M[1]
            dmzdt = self.sample.gamma*np.cross(self.M,self.B)[2] - self.R1*(self.M[2]-self.M0)
        
        dM = np.array([dmxdt, dmydt, dmzdt])

        return dM

    ## This function computed the dext value of M after dt passes
    def compute_next_state(self) -> np.array:
        self.M += self.calculate_bloch_differentials() * self.dt
        return self.M
    
    def wait_on_pulse(self):
        while self.pulse.pulse_start + self.pulse.pulse_duration >= self.renderer.t:
            time.sleep(0)
            

        # if the pulse is over -> shut off the rf magnetic field

        with self.B_eff_mutex:
            self.B_eff = self.B_device.copy()


        with self.pulse_mutex: ## Begin critical section
            self.pulse.pulse_onging = False

        cprint(f"rf pulse shut off at t={self.renderer.t}")

        return

    def pulse_system(self, duration, field:np.array=np.array([])):
        # send in a pulse
        if field.any():
            with self.B_eff_mutex: ## Begin critcal section
                self.B_eff = field.copy()

        else:
            random_vec = np.array(np.random.rand(3))
            random_vec_norm = random_vec/magnitude(random_vec)

            with self.B_eff_mutex: ## Begin critcal section
                self.B_eff = random_vec_norm * 6

        
            
        
        with self.pulse_mutex: ## Begin critcal section
            self.pulse.pulse_start = self.renderer.t
            self.pulse.pulse_duration = duration
            self.pulse.pulse_ongoing = True
        ## End critcal section


        cprint(f"rf pulse is being deployed:\n"\
                f"     will have effective magnetic field {self.B_eff}\n"\
                f"     starting at t={self.pulse.pulse_start}\n"\
                f"     lasting for t={self.pulse.pulse_duration}")

        t2 = threading.Thread(target=self.wait_on_pulse, daemon=True)
        t2.start()

        return

class Graphics_Renderer:
### Simultation Configuration ###

    def __init__(self, simulator):
        self.simulator = simulator
        

        self.ORIGIN = np.array([0,0,0]) # Origin of our world
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(800, 600), show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'  # interactive orbit camera
   
        self.scatter = scene.visuals.Markers(parent=self.view.scene)

        # track the location of M
        self.vector = scene.visuals.Arrow(
                pos=np.vstack([self.ORIGIN, (self.ORIGIN + self.simulator.M)]), 
                width=2,
                color='cyan',
                parent=self.view.scene
            )

        # show the constant B field
        self.B_field = scene.visuals.Arrow(
                pos=np.vstack([self.ORIGIN, (self.ORIGIN + self.simulator.B_eff)/magnitude(self.ORIGIN+self.simulator.B_eff)]), #Normalize the B Field Bector
                width=2,
                color='red',
                parent=self.view.scene
            )

        self.time_label = scene.visuals.Text(
            text='t = 0.0 s',
            color='white',
            font_size=14,
            anchor_x='left',
            anchor_y='top',
            pos=(40, 40),
            parent=self.canvas.scene  # attach to the scene
        )

        self.plot_point = 0
        self.PLOT_RATE = 2
        self.points = []

        self.t = 0

        timer = app.Timer(interval='auto', connect=self.update, start=True)
        simulator.set_renderer(self)



    # Animation function — updates every frame
    def update(self, event):
        global name
        ## Check Syncronization Flags
        if shutdown.is_set():
            app.quit()
            ## Should not get bellow this line
            sys.exit(0)

        if save_next_frame.is_set():
            frame = self.simulator.canvas.render()
            with name_mutex:
                io.write_png(name, frame)
            save_next_frame.clear()
            cprint(f"Success: saved frame to {name}\n")
        
        if not run_sim.is_set():
            return

        t = event.elapsed
        self.time_label.text = f"t = {t} s"
        # advance our simulation
        self.simulator.compute_next_state()

        self.plot_point += 1
        self.plot_point %= self.PLOT_RATE
        self.PLOT_RATE = 2
        if self.plot_point == 0:
            self.points.append((self.simulator.M[0],self.simulator.M[1],self.simulator.M[2])) 
            plot = np.array(self.points)
            self.scatter.set_data(plot, face_color=(0.3,1,1), size=3)

        self.vector.set_data(np.vstack([self.ORIGIN, self.ORIGIN + self.simulator.M]))

        return


class NMR_Console:
    def __init__(self, simulator:Bloch_Simulator, renderer:Graphics_Renderer):
        self.simulator = simulator
        self.renderer = renderer

    def user_thread(self):
        global name

        while True:
            if shutdown.is_set():
                return
            
            with console_mutex:
                args = input("\nNMR cmd >> ")

            # parse commands
            args = args.split(" ")

            if args[0] == "start":
                if run_sim.is_set():
                    cprint("ERR: simulation already running...\n")
                    continue
                run_sim.set()
                cprint("Starting simulation...\n")
                continue

            if args[0] == "config":
                if run_sim.is_set():
                    cprint("ERR: simulation already running...\n")
                    continue

                if len(args) == 1:
                    cprint("Err: incorrect syntax. Try config (one of: B_DEV <Bx> <By> <Bz> or M <Bx> <By> <Bz> or T1 <value> or T1 <value>)")
                    continue

                if args[1] == "B_DEV":
                    m_args = np.array([float(x) for x in args[2:]])
                    self.simulator.B_device = m_args.copy()
                    self.simulator.B_eff = self.simulator.B_device.copy()
                    self.renderer.B_field.set_data(np.vstack([self.renderer.ORIGIN, self.renderer.ORIGIN + self.simulator.B_device]))
                    cprint(f"set B_device to {self.simulator.B_device}")
                    continue

                if args[1] == "M":
                    m_args = np.array([float(x) for x in args[2:]])
                    M = m_args.copy()
                    M0 = magnitude(M)
                    self.renderer.vector.set_data(np.vstack([self.renderer.ORIGIN, self.renderer.ORIGIN + self.simulator.M]))
                    cprint(f"set M to {self.simulator.M}, M0 to {self.simulator.M0}")
                    continue

                if args[1] == "T1":
                    self.simulator.sample.T1 = float(args[2])
                    self.simulator.sample.R1 = 1/self.simulator.sample.T1
                    cprint(f"set T1 to {self.simulator.sample.T1}, R1 to {self.simulator.sample.R1}")
                    continue

                if args[1] == "T2":
                    self.simulator.sample.T2 = float(args[2])
                    self.simulator.sample.R2 = 1/self.simulator.sample.T2
                    cprint(f"set T2 to {self.simulator.sample.T2}, R2 to {self.simulator.sample.R2}")
                    continue

                cprint("Err: incorrect syntax. Try config (one of: B_DEV <Bx> <By> <Bz> or M <Bx> <By> <Bz> or T1 <value> or T1 <value>)")
                continue

            if args[0]=="pulse":
                if not run_sim.is_set():
                    cprint("ERR: simulation is not running...\n")
                    continue
                if self.simulator.pulse.pulse_onging:
                    cprint("ERR: pulse in progress\n")
                    continue

                if len(args) == 1:
                    cprint("ERR: incorrect syntax. Try pulse <duration> or pulse <diration> <Bx> <By> <Bz>")
                    continue

                pulse_args = [float(x) for x in args[1:]]
                if len(pulse_args) == 1:
                    self.simulator.pulse_system(pulse_args[0])
                    continue

                if len(pulse_args) == 4:
                    rf_field = np.array(pulse_args[1:])
                    self.simulator.pulse_system(pulse_args[0], rf_field)
                    continue

                cprint("ERR: incorrect syntax. Try pulse <duration> or pulse <diration> <Bx> <By> <Bz>")
                continue

            if args[0] == "exit":
                shutdown.set()
                continue

            if args[0] == "capture":
                if len(args) == 1:
                    cprint("ERR: incorrect syntax. Try capture filename")
                    continue

                with name_mutex:
                    name = args[1]
                save_next_frame.set()
                

                continue

            if args[0] == "help" or args[0] == "h":
                cprint( "----------------Help for AbelNMR System-----------------\n"\
                        "Version: 1.0.1 (2025)\n"\
                        "Developed by Abel Bellows for CHEM 10\n"\
                        "Commands:\n"
                        "     config: configure device variables (run config for more info)\n"
                        "     start: start the simulation\n"\
                        "     pulse: initiate an RF pulse into the system (run pulse for more info)\n"\
                        "     capture: capture a frame of the simulation (run capture for more info)\n"\
                        "     exit: exit the simulation\n"\
                        "     help/h: get help\n"\
                        "--------------------------------------------------------\n")
                continue
            
            cprint("the command you entered is not a valid command. for help type h\n")
                
    def start(self):
        t1 = threading.Thread(target=self.user_thread, daemon=True)
        t1.start()
        self.renderer.canvas.app.run()


if __name__ == '__main__':
    simulator = Bloch_Simulator()
    renderer = Graphics_Renderer(simulator)
    
    console = NMR_Console(simulator,renderer)

    console.start()
    
