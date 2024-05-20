from dataloader import *


def main():

    data_path = "./data/train_wiki.json"

    dataloader = fewrel_dataloader(data_path)
    print("\nData Loader complete")



if __name__ == "__main__":
    main()