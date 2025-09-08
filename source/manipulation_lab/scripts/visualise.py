from manipulation_lab.scripts.dataset.reader import DatasetReader
import matplotlib.pyplot as plt

def main():
    reader = DatasetReader(
        dataset_dirs=['/home/ubuntu/Projects/manipulation_lab/datasets/room/push_blocks/dagger/']
    )

    print(reader.describe_structure())
    episode = reader.load_episode(0)

    #snapshot = episode["observations"]["sensors"]["scene_camera"]["depth"][10]

    # plt.imshow(snapshot)
    # plt.axis('off')

    # plt.savefig(
    #     '/home/ubuntu/Projects/manipulation_lab/outputs/snapshot.png', 
    #     bbox_inches='tight', 
    #     pad_inches=0
    # )

if __name__ == "__main__":
    main()