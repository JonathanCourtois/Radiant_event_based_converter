# Radiant_EVB_Model
Give an event based camera converter for RGB Video.

## How to use : 

### Python env information :
Install the project with uv :

If you haven't uv :
    pip install uv
then
    - Clone the project
    - uv sync
    - Run the uv Command in Example


### Useage : 

python ./video_converter.py [-h] [--output_path OUTPUT_PATH] [--threshold THRESHOLD] [--noise_level NOISE_LEVEL] [--merge_method MERGE_METHOD] [--fps FPS] [--change_input_rate] [--input_rate INPUT_RATE] [--save_spike_mat]
                          video_path

Convert RGB video to Event Based video.

positional arguments:
  video_path            Path to the video to convert.

optional arguments:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Path to the output video.
  --threshold THRESHOLD
                        Threshold for the event camera.
  --noise_level NOISE_LEVEL
                        Noise level for the event camera.
  --merge_method MERGE_METHOD
                        Method to merge the channels to RGB. Can Be "first_channel", "polarity" or "channel"
  --fps FPS             Fps of the output video.
  --change_input_rate   Change the input rate of the video.
  --input_rate INPUT_RATE
                        Input rate of the video (convert every n frame).
  --save_spike_mat      Save the spike matrix as npy file.

### Example : 

    uv run ./video_converter.py ./video_sources/video_name.mp4 --merge_method channel --save_spike_mat
or
    uv run ./video_converter.py ./video_sources/video_name.mp4 --merge_method channel --fps 8 --change_input_rate --input_rate 20

## Citation : 
Every work that use this Event Based Converter Model Camera must reference it with : 
@article{courtois2025spiking,
  title={Spiking monocular event based 6D pose estimation for space application},
  author={Courtois, Jonathan and Miramond, Beno{\^\i}t and Pegatoquet, Alain},
  journal={arXiv preprint arXiv:2501.02916},
  year={2025}
}
MLA:
Courtois, Jonathan, Benoît Miramond, and Alain Pegatoquet. "Spiking monocular event based 6D pose estimation for space application." arXiv preprint arXiv:2501.02916 (2025).

