# NMR Machine Simulator

## Purpose
To demonstrate the macroscopic behavior of M in an NMR machine and to make clear the workflow of NMR analysis, I constructed a simulator of Bloch Equations. The user can configure all aspects of the device including B_DEV, M, T1, T2, and can send RF pulses with a specific applied magnetic field. The NMR simulator relies on the Vispy open source graphics library.

## Usage
### Python Enviroment Configuration
To setup the python enviroment, first create your venv by running `python3 -m venv venv` then activate your venv by running `source venv/bin/activate`. Install all of the required packages by running `pip3 install -r requirements.txt`.

### Start the program
To run the simulator run `python3 NMR_SIM.py`. You may want to use a screen split function so that you can see all 4 windows at one time.

### Configure simulation 
To configure enromental variables of the simulator, use the `config` command in the command console. You can configure the magnetic field of the device (`B_DEV`), T1 and T2 (`T1` and `T2`), and net starting magnetization (`M`) with the following commands

1. `config B_DEV <Bx> <By> <Bz>`
2. `config M <Bx> <By> <Bz>`
3. `config T1 <value>`
4. `config T2 <value>`


### Start the simulation
You can start the simulation by running `start`. You can pause the simulation by running `pause`.

### Initiating radio pulse
You can initiate a radio pule with the following commands: `pulse <duration>` or `pulse <duration> <Bx> <By> <Bz>`

### Capture a photo of the simulation
You can capture a photo of the current frame by running `capture <filename>.png`

### Fourier Transform to see the final NMR spectrum
Viewing the FID graph run `fft <t1> <t2>` to initiate an fourier transform with the data between t=t1 and t=t2.

## Model Noise
To model a noisy enviroment run `sh noisy.sh <pulse_timing>` to snd in short radio bursts every `<pulse_timing>` seconds.