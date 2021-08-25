from typing import Optional


class Node(object):
    __slots__ = [
        "num",
        "next"
    ]

    def __init__(self, num: float) -> None:
        self.num = num
        self.next: Optional[Node] = None


class MAB_Nodes(object):
    def __init__(self) -> None:
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.__len = 0

    def add(self, num: float):
        if self.head is None:
            self.head = Node(num)
            self.tail = self.head
        else:
            self.tail.next = Node(num)
            self.tail = self.tail.next
        self.__len += 1

    def run(self):
        p = self.head
        while p is not None:
            yield p.num
            p = p.next

    def avg(self) -> float:
        ss = 0
        for n in self.run():
            ss += n
        return ss/self.__len

    def __len__(self):
        return self.__len
