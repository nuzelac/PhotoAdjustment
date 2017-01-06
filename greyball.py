from tests.tests import *

def main():
    # Test1()

    # Test2()

    # Test3()

    # Test4()

    # Test5("gb", 30, 1000, 128, 0.001)

    # Test6("gb", 30, 1000, 64, 0.001)

    # Test7("gb", 50, 100, 64, 0.01, "s8_s4_s2_l1", "tmp.txt")

    # Test8()

    # Test9("tasks.sh", "results/", 5, "./task_", ".sh")

    # Test7("lgb", 30, 10, 64, 0.01, "s8_s1")

    # Test10("lgb", 30, 200, 64, 0.001, "s2_s2_s2_s2_s1", 0, 0, 0, 0)

    # Test11("tasks.sh", "results0/", 3, "./task_", ".sh", 0, 0)

    # Test13("results0/lgb*.txt", "lgb")

    # Test13("results0/gb_30_10_64_0.01_s8_s1.txt", "gb")

    # Test14("lgb", 30, 50, 64, 0.01, "s8_s1", -1, 1, -1, 1, 10, None)

    # Test15("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 3, None)

    # Test16("tasks.sh", "results_r/", 3, "./task_", ".sh", -1, 1, -1, 1, 11, 20)

    # Test13("results_r/lgb*s2_s2_s2_s1*.txt", "lgb")

    # smaller max. error
    # Test7("lgb", 30, 10, 16, 0.01, "s4_s4_s1")

    # Test7("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s2_s1")

    # OK
    # Test7("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1")

    # Test17("lgb", 10, 30, 64, 0.01, "s2_s2_s2_s1")

    # Test15("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # Test18("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1")

    # something seems to be wrong here...
    # Test19("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # Test20("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # Test21("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # Test22("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # super
    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)

    # Test23("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 1)

    # Test24("lgb", 30, 20, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)

    # Test25("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 1)

    # Test26("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5)

    # Test27("lgb", 30, 500, 16, 0.001, "s8_s1", -1, 1, -1, 1, 0, 14, 13)

    # Test28("lgb", 30, 500, 16, 0.001, "s8_s1", -1, 1, -1, 1, 0, 14, 13)

    # Test29("lgb", 30, 10, 64, 0.01, "s8_s1")

    # Test7("lgb", 30, 10, 64, 0.01, "s8_s1")

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)

    # Test30("gb", 30, 10, 64, 0.01, "s8_s1")

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 25)

    # Test31("lgb", 30, 50, 64, 0.001, "s8_s1", -1, 1, -1, 1, 5)

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)

    # same as Test22, but with a reduced training set after the selection of the best initial values
    # Test32("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)

    # Test32("lgb", 30, 10, 64, 0.005, "s8_s1", -1, 1, -1, 1, 3)

    # same as Test31, but with a reduced training set after the selection of the best initial values
    # Test33("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)

    # Test32("gb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5)
    # Test32("gb", 30, 10, 64, 0.01, "r8_r5_s1", -1, 1, -1, 1, 5)

    # Test32("lgb", 30, 10, 64, 0.012, "s8_s1", -1, 1, -1, 1, 5)

    # Test35("lgb", 30, 10, 64, 0.012, "r8_r8_s1", -1, 1, -1, 1, 5)
    Test35("lgb", 20, 10, 64, 0.012, "r5_r5_s1", -1, 1, -1, 1, 5)

    pass


if (__name__ == "__main__"):
    main()
