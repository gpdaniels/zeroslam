# ZeroSLAM #

SLAM without dependencies.

```
.-----------------------------------------------.
|   _____             _____ __    _____ _____   |
|  |__   |___ ___ ___|   __|  |  |  _  |     |  |
|  |   __| -_|  _| . |__   |  |__|     | | | |  |
|  |_____|___|_| |___|_____|_____|__|__|_|_|_|  |
|                                               |
| This software is a:                           |
|  |- simple                                    |
|  |- minimal                                   |
|  |- indirect                                  |
|  |- monocular                                 |
|  |- factor-graph                              |
|  |- deterministic                             |
|  |- dependency-free                           |
|  '- visual SLAM system written in pure C++.   |
|                                               |
| No external libraries. No frills. Just SLAM.  |
|                                               |
| >   https://github.com/gpdaniels/zeroslam   < |
|                                               |
| Licensed under GPLv3                          |
| Get in touch for commercial licensing.        |
'-----------------------------------------------'
```

## Building and testing ##

Standard cmake workflow:

```
mkdir -p build
cd build
cmake ..
cmake --build . --parallel 4
ctest
```

## Processing a video file ##

First extract the frames as pgm images with file names starting at `000.pgm`.
This can be done easily with ffmpeg, e.g. for the file `video.mp4`:
```
cd build
mkdir -p data
ffmpeg -i video.mp4 -start_number 0 -frames:v 10 'data/%03d.pgm'
```

Next run the system on the data. The usage of the program is `zeroslam [video] [fx] [fy] [cx] [cy]`
```
cd build
./runtime/Release/zeroslam ./data 525 525 320 240
```

The program will output a trajectory file in TUM format and a ply pointcloud file.

## Evaluating a trajectory ##

The tools directory contains a tool called `trajectory_alignment`.
This tool aligns trajectories in the TUM format and returns the error after scaling and alignment.

Usage:
```
# Build the tool.
cd build
cmake --build . --parallel 4 --target trajectory_alignment

# Evaluate a trajectory with a ground truth.
./runtime/Release/trajectory_alignment trajectory_gt.txt trajectory_eval_1.txt

# Evaluate two trajectories against a ground truth.
./runtime/Release/trajectory_alignment trajectory_gt.txt trajectory_eval_1.txt trajectory_eval_2.txt

# Ensure the first pose is aligned.
./runtime/Release/trajectory_alignment trajectory_gt.txt trajectory_eval_1.txt --first

# Plot the trajectories from each of the x, y, or z, planes.
./runtime/Release/trajectory_alignment trajectory_gt.txt trajectory_eval_1.txt --plot xyz
```

## License ##

Copyright (C) 2026 Geoffrey Daniels. https://gpdaniels.com/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
