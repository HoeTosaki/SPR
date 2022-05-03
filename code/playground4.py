from multiprocessing import Process
import multiprocessing
import time


def t():
    while True:
        print("----test---")
        time.sleep(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    p = Process(target=t)
    p.start()

    while True:
        print("----main---")
        time.sleep(3)
