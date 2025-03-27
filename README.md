# Radiant EVB Model

An event-based camera converter for RGB video.

## How to Use

### Python Environment Setup

To install the project with `uv`:

1. If you don't have `uv`, install it using:
  ```bash
  pip install uv
  ```
2. Clone the project repository.
3. Run:
  ```bash
  uv sync
  ```
4. Use the `uv` command as shown in the examples below.

### Usage

Run the following command to convert an RGB video to an event-based video:

```bash
python ./video_converter.py [-h] [--output_path OUTPUT_PATH] [--threshold THRESHOLD] [--noise_level NOISE_LEVEL] [--merge_method MERGE_METHOD] [--fps FPS] [--change_input_rate] [--input_rate INPUT_RATE] [--save_spike_mat] video_path
```

#### Positional Arguments:
- `video_path`: Path to the video to convert.

#### Optional Arguments:
- `-h, --help`: Show the help message and exit.
- `--output_path OUTPUT_PATH`: Path to the output video.
- `--threshold THRESHOLD`: Threshold for the event camera.
- `--noise_level NOISE_LEVEL`: Noise level for the event camera.
- `--merge_method MERGE_METHOD`: Method to merge the channels to RGB. Options: `"first_channel"`, `"polarity"`, or `"channel"`.
- `--fps FPS`: Frames per second of the output video.
- `--change_input_rate`: Change the input rate of the video.
- `--input_rate INPUT_RATE`: Input rate of the video (convert every n frames).
- `--save_spike_mat`: Save the spike matrix as an `.npy` file.

### Examples

```bash
uv run ./video_converter.py ./video_sources/video_name.mp4 --merge_method channel --save_spike_mat
```

```bash
uv run ./video_converter.py ./video_sources/video_name.mp4 --merge_method channel --fps 8 --change_input_rate --input_rate 20
```

## Citation

If you use this Event-Based Converter Model Camera in your work, please cite it as follows:

### BibTeX Citation:
```bibtex
@article{courtois2025spiking,
  title={Spiking monocular event based 6D pose estimation for space application},
  author={Courtois, Jonathan and Miramond, Benoît and Pegatoquet, Alain},
  journal={arXiv preprint arXiv:2501.02916},
  year={2025}
}
```

### MLA Citation:
Courtois, Jonathan, Benoît Miramond, and Alain Pegatoquet. "Spiking monocular event based 6D pose estimation for space application." *arXiv preprint arXiv:2501.02916* (2025).
