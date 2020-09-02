"""The main program to call"""
# from particulates import PM25
from weather import Forcast
# from classification import Income

if __name__ == "__main__":
    AGENT = Forcast()
    AGENT.read_train()
    AGENT.read_test()
    AGENT.train_model()
    AGENT.draw('train', 'train.png')
    AGENT.draw('test', 'test.png')
    # AGENT = Income()
    # AGENT.encode('classification/simple.csv')
    # AGENT.training()
