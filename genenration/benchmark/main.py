from connectivity_ import connect_datasets_generation
from cycle_ import  cycle_datasets_generation
from shortest import shortest_datasets_generation
from utils import load_yaml, build_args, write_to_file
import os


def main():
    task_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "task_config_test.yaml"))
    args = build_args()
    task_list=['cycle_train', 'connectivity_train', 'shortest_train']
    for key in task_list:
        task_config = task_config_lookup[key]
        print(f"Generating dataset for task: {key}")
        print(task_config)
        if "cycle" in key:
            cycle_datasets_generation(task_config)
        elif "connectivity" in key:
            connect_datasets_generation(task_config)
        elif "shortest" in key:
            shortest_datasets_generation(task_config)


if __name__ == "__main__":
    main()