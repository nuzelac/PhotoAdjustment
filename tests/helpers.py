import numpy as np

def getTestFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346]

    start = 0
    if (idx > 0):
        start = ends[idx - 1]
    end = ends[idx]

    return data[start:end]


def getTrainFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346]

    start = 0
    if (idx > 0):
        start = ends[idx - 1]
    end = ends[idx]

    take = data[:]
    take = np.delete(take, list(range(start, end)), axis=0)

    return take


def getFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346]

    start = 0
    if (idx > 0):
        start = ends[idx - 1]
    end = ends[idx]

    return data[start:end]


def getFolds(data, foldsIndices):
    take = getFold(data, foldsIndices[0])

    for i in range(1, len(foldsIndices)):
        idx = foldsIndices[i]
        take = np.concatenate((take, getFold(data, idx)), axis=0)

    return take


def getSubTrainFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346]

    remove = idx
    remove2 = idx - 1
    if (remove2 < 0):
        remove2 = 14

    indices = []
    for i in range(15):
        if (i != remove and i != remove2):
            indices.append(i)

    take = getFold(data, indices[0])

    for i in indices[1:]:
        take = np.concatenate((take, getFold(data, i)), axis=0)

    return take


def getSubTestFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346]

    idx -= 1
    if (idx < 0):
        idx = 14

    take = getFold(data, idx)

    return take


def loadLines(path):
    with open(path, "r") as f:
        lines = [l.strip("\r\n") for l in f.readlines()]

    return lines


def loadPlainMatrix(path):
    lines = loadLines(path)

    gt = []
    for l in lines:
        parts = l.split(" ")
        current = []
        for p in parts:
            if (p != ""):
                current.append(float(p))
        gt.append(current)

    return np.array(gt)


def dotProduct(v1, v2):
    return np.sum(np.multiply(v1, v2))


def calculateAngle(v1, v2):
    return 180.0 * np.arccos(dotProduct(v1, v2) / np.sqrt(dotProduct(v1, v1) * dotProduct(v2, v2))) / np.pi


def calculateAngularStatistics(angles1, angles2):
    n = angles1.shape[0]

    angles = []
    for i in range(n):
        angles.append(calculateAngle(angles1[i, :], angles2[i, :]))

    return (np.mean(angles), np.median(angles), np.max(angles))


def getBatches(data, batchSize=64):
    from math import ceil

    n = int(ceil(float(len(data)) / batchSize))

    batches = []

    for i in range(n):
        currentIdx = i * batchSize
        nextIdx = (i + 1) * batchSize
        batches.append(data[currentIdx:nextIdx, :])

    return batches
