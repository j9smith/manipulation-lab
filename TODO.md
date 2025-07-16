## Project-wide
- [] Implement W&B logging
- [] Centralise processing of images and sensor data - we're currently tackling them separately during training and inference, which could risk divergence.
- [] **Critical**: Discrepency between control frequency and data collection frequency. How do we train a policy that operates at, e.g., 10Hz on data that is available at 30Hz? Do we scale deltas? 

## Simulation
- [X] Enable simulation resets without having to reload the app (same as env.reset(), which doesn't work)

## Scene Design
- [] Specify 'rigs' that are importable across scenes, instead of hard-coding manipulator + cameras.

## Controller
- [X] Populate init config via hydra
- [] Implement _schedule_actions for actions chunks etc.
- [X] Implement control _step logic.
- [] How do we handle sensor data?

## Runner
- [] Gracefully handle end of episodes.

## TeleopHandler
- [X] Gracefully handle end of episodes.
- [] Fix bug: socket listener thread starts even when controller not connected, leading to malformed first byte and critical error **(haven't been able to reproduce)**. 

## DatasetReader
- [] Optimise loading episodes from dataset

## DatasetWrapper
- [] Optimise \__getitem__
- [] Episode cache might be problematic where many/long episodes are present (all episodes will be stored in memory). Optimise this. This also leads to OOM crashes when using dataloaders with num_workers > 0. 
- [X] Always return structured obs, and we'll handle it with a custom collate function

## Data Loaders
- [X] Batch processing of images instead, sequential is too slow

## Train
- [] Add validation data and then measure accuracy/loss against val to better inform training metrics
- [] Add open loop roll-outs for debugging

## Models - Policy 
- [] Add a network that processes the visual embedding dims separate from the proprio dims, and then fuse them

## Models - Encoder
- []

## Tasks
- []