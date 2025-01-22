from enum import Enum


class A:

    def name(self):
        print('A')

class B:

    def name(self):
        print('B')

class PrintType(Enum):
    A = A
    B = B
    
class PrintContext:
    def __init__(self, value):
        if value in PrintType.__members__:
            self._strategy = PrintType[value].value()
        else:
            raise ValueError(f'value {value} not in PrintType')

    def name(self):
        self._strategy.name()

def example(value):
    r = PrintContext(value)
    r.name()
    pass

if __name__ == '__main__':
    example('A')