from dataloader import Dataloader



def test_dataloader():
    dataloader = Dataloader()
    x,y = dataloader[11]

    print(x)
    print(y)
