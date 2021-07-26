from impact_measure import impact, momentum, cohen_d
import numpy as np


def test_impact():
    x_1 = np.random.random(size=1000) * 10
    x_2 = np.random.random(size=1000) * 15
    assert impact(x_1, x_2) is not None

def debug_cohen_d():
    x_1 = np.random.random(size=1000) * 10
    x_2 = np.random.random(size=1000) * 15
    d = cohen_d(x_1, x_2)
    import code
    code.interact(local=locals())
    
def debug_impact():
    x_1 = np.random.random(size=1000) * 10
    x_2 = np.random.random(size=1000) * 15
    i = impact(x_1, x_2)
    print(i)
    import code
    code.interact(local=locals())

if __name__ == '__main__':
    debug_impact()
