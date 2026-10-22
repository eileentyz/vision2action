# Vision2Action: Cloud-Validated Robotic Manipulation

**From pixels to action: Robotic perception powered by OpenCV 5, with cloud-based evaluation using AWS.**

Developed for the **OpenCV AI Competition 2026, powered by AWS**.

## Demo

![Vision2Action Demo](docs/media/live_mission_opencv5.gif)

[▶ Watch the Demo Video](docs/media/live_mission_opencv5.mp4)

Vision2Action uses **OpenCV 5 to help a simulated robot detect, locate and pick up objects**. It also uses **AWS to evaluate how accurately the vision system works**.

The goal is to explore how computer vision can support autonomous material handling and make robotic perception easier to test.

## Why Vision2Action?

In manufacturing environments, workers often need to pick up and move materials between different locations.

Robots can help automate these tasks, but they first need to know where an object is.

Even a small error in object detection or position estimation can cause a robot to miss its grasp.

Vision2Action brings together **computer vision, robotics and cloud computing** to explore this challenge.

## What Can It Do?

Vision2Action has two main parts.

### 1. OpenCV 5-Powered Robotic Manipulation

The simulated Galbot robot can:

- Navigate to a pickup location.
- Detect and segment a cube using OpenCV 5 and YOLO11.
- Estimate the cube's position and orientation.
- Plan and carry out a grasp.
- Transport the cube to another location and place it.

OpenCV 5 processes the robot's camera images during the mission, so it is part of the live perception system.

### 2. AWS Cloud Evaluation

Vision2Action also provides a way to evaluate the perception results.

After a mission:

1. Evaluation data is uploaded to Amazon S3.
2. AWS Lambda runs OpenCV 5 to process the images.
3. An evaluation report is generated.
4. The results can be compared with those obtained locally.

This helps developers check whether their evaluation results are consistent across different computing environments.

## How It Works?

![Vision2Action System Architecture](docs/figures/fig_live_end_to_end.png)

**Robot perception and manipulation**

Camera → OpenCV 5 + YOLO11 → Object Position → Robot Grasping

**Cloud evaluation**

Evaluation Data → Amazon S3 → AWS Lambda → Evaluation Report

The robot operates locally in MuJoCo simulation. AWS is used to evaluate the recorded perception data, not to control the robot.

### Technologies Used

- **OpenCV 5:** Object detection and segmentation using its DNN module.
- **YOLO11:** Segmentation model exported to ONNX.
- **ROS 2:** Integration of robotic components.
- **Nav2 and MoveIt 2:** Robot navigation and manipulation.
- **MuJoCo:** Robot simulation.
- **AWS S3, Lambda and ECR:** Cloud-based evaluation.
- **Python and Docker:** Development and deployment.

## Results

### OpenCV 5 Performance

The OpenCV 5 implementation was compared with the original Ultralytics pipeline using 150 validation images.

| Metric | Result |
|---|---|
| Validation images | 150 |
| Segmentation masks | Identical to Ultralytics |
| Original median inference time | 25.3 ms |
| OpenCV 5 median inference time | 20.9 ms |
| Inference latency reduction | **17.4%** |

Both implementations produced the same segmentation masks, while OpenCV 5 achieved lower median inference latency on the tested laptop.

### Autonomous Robot Demonstration

Across **five recorded simulated missions**, all five completed successfully.

| Metric | Result |
|---|---|
| Successful missions | **5/5** |
| Final placement error | 3.3–6.1 mm |
| Pick-position estimation error | 1.6–2.9 mm |

The featured demo shows the robot completing a pick-and-place task with a **3.6 mm placement error**.

These results were obtained in simulation.

### AWS Evaluation

The same exported evaluation data was processed locally and on AWS.

| Metric | Local | AWS |
|---|---|---|
| Segmentation IoU | 0.896 | 0.896 |
| Orientation error | 1.27° | 1.27° |

The results matched, showing that the evaluation could be reproduced in the cloud for the tested images.

## Getting Started

The repository includes the **OpenCV 5 perception system, evaluation tools, model, sample data and AWS pipeline**.

The complete ROS 2 robot workspace is part of an ongoing final-year project and is not included. However, the OpenCV 5 inference and evaluation can be tested independently.

### Requirements

- Python 3.12
- Linux or macOS
- Docker (for local AWS Lambda testing)

### 1. Install Dependencies

```bash
git clone https://github.com/eileentyz/vision2action.git
cd vision2action

python3 -m venv .venv-cv5
.venv-cv5/bin/pip install -r requirements-cv5.txt
```

### 2. Run the OpenCV 5 Benchmark

```bash
.venv-cv5/bin/python tools/benchmark_dnn.py \
  --data data/cube_seg \
  --threads 8 \
  --output build/benchmark
```

This runs OpenCV 5 on the validation images and compares the results against the reference outputs.

### 3. Run a Local Evaluation

```bash
.venv-cv5/bin/python tools/make_synthetic_bundle.py

.venv-cv5/bin/python tools/evaluate.py \
  --job-id synthetic-dnn \
  --bundle samples/synthetic/synthetic_stereo.npz \
  --mask-sources dnn,gt,hsv \
  --model models/cube_yolo11n_seg.onnx \
  --threads 8 \
  --output build/synthetic-dnn
```

This generates an evaluation report using OpenCV 5.

No AWS account is needed.

### 4. Test the AWS Lambda Application Locally

```bash
aws/local_smoke_test.sh --with-model
```

This uses Docker to test the Lambda application without deploying anything to AWS.

For the full AWS deployment instructions, see [AWS Setup Guide](aws/README.md).

## Repository Structure

| Folder | Description |
|---|---|
| `galbot_cv5/` | OpenCV 5 inference and evaluation code |
| `integration/ros2/` | Connects OpenCV 5 with the ROS 2 robot |
| `models/` | YOLO11 segmentation model |
| `data/` | Validation images and labels |
| `tools/` | Benchmark and evaluation scripts |
| `aws/` | AWS Lambda application and deployment setup |
| `results/` | Performance results and evaluation reports |
| `docs/` | Technical report, diagrams and demo videos |
| `tests/` | Automated tests |

## Documentation

For more details:

- [Technical Report](docs/TECHNICAL_REPORT.md)
- [AWS Setup Guide](aws/README.md)
- [Live Robot Demo Guide](live/README.md)
- [ROS 2 Integration](integration/ros2/README.md)
- [Evaluation Results](results/)
- [Model Information](models/README.md)

## Limitations

Vision2Action is currently tested in simulation.

The robot demonstrations use a cube in a controlled environment. The five recorded missions use the same cube rotation, so further testing is needed to understand how well the system performs in different situations.

The complete FYP robot workspace is not included in this repository. The OpenCV 5 inference, benchmarks and evaluation tools can be reproduced independently, but the full robot mission requires that separate workspace.

The AWS evaluation pipeline is also deployed privately and is not currently a public service.

## Future Work

- Test the system on a physical robot.
- Support more object types.
- Improve detection when objects move or become partially hidden.
- Test different lighting conditions and environments.
- Expand the cloud evaluation with more datasets and test cases.

## Licence and Acknowledgements

This project is released under the [GNU AGPL v3.0](LICENSE).

Built using OpenCV 5, YOLO11, ROS 2, MuJoCo and AWS.

Developed as a competition-focused extension of an ongoing robotics project.

See [NOTICE.md](NOTICE.md) for third-party licences and acknowledgements.
