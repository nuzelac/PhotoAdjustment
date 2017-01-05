import numpy as np;
import tensorflow as tf;


def getTestFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346];

    start = 0;
    if (idx > 0):
        start = ends[idx - 1];
    end = ends[idx];

    return data[start:end];


def getTrainFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346];

    start = 0;
    if (idx > 0):
        start = ends[idx - 1];
    end = ends[idx];

    take = data[:];
    take = np.delete(take, range(start, end), axis=0);

    return take;


def getFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346];

    start = 0;
    if (idx > 0):
        start = ends[idx - 1];
    end = ends[idx];

    return data[start:end];


def getFolds(data, foldsIndices):
    take = getFold(data, foldsIndices[0]);

    for i in range(1, len(foldsIndices)):
        idx = foldsIndices[i];
        take = np.concatenate((take, getFold(data, idx)), axis=0);

    return take;


def getSubTrainFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346];

    remove = idx;
    remove2 = idx - 1;
    if (remove2 < 0):
        remove2 = 14;

    indices = [];
    for i in range(15):
        if (i != remove and i != remove2):
            indices.append(i);

    take = getFold(data, indices[0]);

    for i in indices[1:]:
        take = np.concatenate((take, getFold(data, i)), axis=0);

    return take;


def getSubTestFold(data, idx):
    ends = [1273, 2225, 2501, 3482, 3888, 4387, 5343, 6051, 6606, 7702, 8215, 9527, 10068, 11265, 11346];

    idx -= 1;
    if (idx < 0):
        idx = 14;

    take = getFold(data, idx);

    return take;


def loadLines(path):
    with open(path, "r") as f:
        lines = [l.strip("\r\n") for l in f.readlines()];

    return lines;


def loadPlainMatrix(path):
    lines = loadLines(path);

    gt = [];
    for l in lines:
        parts = l.split(" ");
        current = [];
        for p in parts:
            if (p != ""):
                current.append(float(p));
        gt.append(current);

    return np.array(gt);


def dotProduct(v1, v2):
    return np.sum(np.multiply(v1, v2));


def calculateAngle(v1, v2):
    return 180.0 * np.arccos(dotProduct(v1, v2) / np.sqrt(dotProduct(v1, v1) * dotProduct(v2, v2))) / np.pi;


def calculateAngularStatistics(angles1, angles2):
    n = angles1.shape[0];

    angles = [];
    for i in range(n):
        angles.append(calculateAngle(angles1[i, :], angles2[i, :]));

    return (np.mean(angles), np.median(angles), np.max(angles));


def getBatches(data, batchSize=64):
    from math import ceil;

    n = int(ceil(float(len(data)) / batchSize));

    batches = [];

    for i in range(n):
        currentIdx = i * batchSize;
        nextIdx = (i + 1) * batchSize;
        batches.append(data[currentIdx:nextIdx, :]);

    return batches;


def Test1():
    gt = loadPlainMatrix("gb.txt");

    p = np.polyfit(gt[:, 0], gt[:, 2], 1);

    print(p);


def Test2():
    gt = loadPlainMatrix("gb.txt");

    s = 0;
    for i in range(15):
        s += 11346 - len(getTrainFold(gt, i));
    print(s);


def Test3():
    gt = loadPlainMatrix("gb.txt");

    p = np.polyfit(gt[:, 0], gt[:, 2], 1);

    ie = np.copy(gt);

    for i in range(gt.shape[0]):
        r = gt[i, 0];
        r2 = r + np.random.normal(0, 0.05);
        b2 = p[0] * r2 + p[1];
        g2 = 1 - r2 - b2;
        ie[i, :] = [r2, g2, b2];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);


def Test4():
    gt = loadPlainMatrix("gb.txt");

    batches = getBatches(gt, 64);

    print(len(batches[0]));
    print(len(batches[-1]));


def Test5(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01):
    print(
        "Dataset: " + prefix + ", epochs: " + str(epochs) + ", batchSize: " + str(batchSize) + ", learningRate: " + str(
            learningRate));

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        w1 = tf.Variable(tf.random_uniform([1, inputN], -1, 1));
        b1 = tf.Variable(tf.random_uniform([1, 1], -1, 1));
        linear1 = tf.add(tf.matmul(w1, x), b1);

        finalOutput = linear1;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);


def Test6(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01):
    print(
        "Dataset: " + prefix + ", epochs: " + str(epochs) + ", batchSize: " + str(batchSize) + ", learningRate: " + str(
            learningRate));

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        sigmoid1N = 20;

        w1 = tf.Variable(tf.random_uniform([sigmoid1N, inputN], -1, 1));
        b1 = tf.Variable(tf.random_uniform([sigmoid1N, 1], -1, 1));
        sigmoid1 = tf.sigmoid(tf.add(tf.matmul(w1, x), b1));

        w2 = tf.Variable(tf.random_uniform([1, sigmoid1N], -1, 1));
        b2 = tf.Variable(tf.random_uniform([1, 1], -1, 1));
        linear1 = tf.add(tf.matmul(w2, sigmoid1), b2);

        finalOutput = linear1;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);


def Test7(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration);

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], -1, 1));
            bias = tf.Variable(tf.random_uniform([n, 1], -1, 1));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test8():
    from sys import argv;

    args = argv[1:];

    if (len(args) != 7):
        return;

    Test7(args[0], int(args[1]), int(args[2]), int(args[3]), float(args[4]), args[5], args[6]);


def Test9(mainOutputPath, resultsPrefix, tasksN, taskPrefix, taskSuffix):
    ns = [20, 30, 40, 50];
    es = [100, 1000];
    learningRates = [0.02, 0.01, 0.005, 0.002, 0.001];
    bss = [16, 64, 256];
    ds = ["gb", "lgb"];
    confs = ["l1", "s4_l1", "s8_l1", "s16_l1", "r4_l1", "r8_l1", "r16_l1", "s16_s8_l1", "s16_s8_s4_l1", "s1", "s4_s1",
             "s8_s1", "s16_s1", "r4_s1", "r8_s1", "r16_s1", "s16_s8_s1", "s16_s8_s4_s1"];

    tasks = [];
    for dataset in ds:
        for n in ns:
            for epochs in es:
                for batchSize in bss:
                    for learningRate in learningRates:
                        for configuration in confs:
                            outputPath = resultsPrefix + dataset + "_" + str(n) + "_" + str(epochs) + "_" + str(
                                batchSize) + "_" + str(learningRate) + "_" + configuration + ".txt";
                            tasks.append("python task.py " + dataset + " " + str(n) + " " + str(epochs) + " " + str(
                                batchSize) + " " + str(learningRate) + " " + configuration + " " + outputPath);

    separateTasks = [];
    for i in range(tasksN):
        separateTasks.append([]);

    current = 0;
    for i in range(len(tasks)):
        separateTasks[i % tasksN].append(tasks[i]);

    for i in range(tasksN):
        with open(taskPrefix + str(i) + taskSuffix, "w") as f:
            for task in separateTasks[i]:
                f.write(task + "\n");

    with open(mainOutputPath, "w") as f:
        for i in range(tasksN):
            f.write(taskPrefix + str(i) + taskSuffix + " &\n");


def Test10(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", outputPath=None,
           lowerW=-1, upperW=1, lowerB=-1, upperB=1):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
            bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test11(mainOutputPath, resultsPrefix, tasksN, taskPrefix, taskSuffix, initialW, initialB):
    ns = [20, 30, 40, 50];
    es = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    learningRates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01];
    bss = [64];
    ds = ["gb", "lgb"];
    confs = ["s8_s1"];

    tasks = [];
    for dataset in ds:
        for n in ns:
            for epochs in es:
                for batchSize in bss:
                    for learningRate in learningRates:
                        for configuration in confs:
                            outputPath = resultsPrefix + dataset + "_" + str(n) + "_" + str(epochs) + "_" + str(
                                batchSize) + "_" + str(learningRate) + "_" + configuration + ".txt";
                            tasks.append("python task.py " + dataset + " " + str(n) + " " + str(epochs) + " " + str(
                                batchSize) + " " + str(
                                learningRate) + " " + configuration + " " + outputPath + " " + str(
                                initialW) + " " + str(initialB));

    separateTasks = [];
    for i in range(tasksN):
        separateTasks.append([]);

    current = 0;
    for i in range(len(tasks)):
        separateTasks[i % tasksN].append(tasks[i]);

    for i in range(tasksN):
        with open(taskPrefix + str(i) + taskSuffix, "w") as f:
            for task in separateTasks[i]:
                f.write(task + "\n");

    with open(mainOutputPath, "w") as f:
        for i in range(tasksN):
            f.write(taskPrefix + str(i) + taskSuffix + " &\n");


def Test12():
    from sys import argv;

    args = argv[1:];

    if (len(args) != 9):
        return;

    Test10(args[0], int(args[1]), int(args[2]), int(args[3]), float(args[4]), args[5], args[6], int(args[7]),
           int(args[7]), int(args[8]), int(args[8]));


# def Test13(pattern, dataset):
    gt = loadPlainMatrix(dataset + ".txt");

    from glob import glob;

    bestMedian = -1;
    for f in glob(pattern):
        ie = loadPlainMatrix(f);
        statistics = calculateAngularStatistics(gt, ie);
        if (bestMedian == -1 or statistics[1] < bestMedian):
            bestMedian = statistics[1];
            print(f);
            print("\t" + str(bestMedian));


def Test14(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
            bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        currentStatistics = [];
        currentResults = [];

        for trial in range(trialsCount):
            print("\tTrial " + str(trial + 1));
            current = np.copy(trainGT);
            currentIE = np.copy(testGT);

            with tf.Session() as session:
                session.run(init);

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(trainGT)):
                    red = session.run(re, feed_dict={x: trainFeatures[j:j + 1, :].transpose(), r: [trainGT[j, 0:1]],
                                                     g: [trainGT[j, 1:2]], b: [trainGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    current[j, :] = [red, green, blue];

                currentStatistics.append(calculateAngularStatistics(trainGT, current));

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    currentIE[j, :] = [red, green, blue];

                currentResults.append(currentIE);

        best = 0;
        for i in range(1, trialsCount):
            if (currentStatistics[i][1] < currentStatistics[best][1]):
                best = i;

        ie[start:start + len(testGT), :] = currentResults[best];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test15(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getSubTrainFold(gt, i);
        trainFeatures = getSubTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);
        subTestGT = getSubTestFold(gt, i);
        subTestFeatures = getSubTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
            bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        currentStatistics = [];
        currentResults = [];

        for trial in range(trialsCount):
            print("\tTrial " + str(trial + 1));
            current = np.copy(subTestGT);
            currentIE = np.copy(testGT);

            with tf.Session() as session:
                session.run(init);

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(subTestGT)):
                    red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(), r: [subTestGT[j, 0:1]],
                                                     g: [subTestGT[j, 1:2]], b: [subTestGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    current[j, :] = [red, green, blue];

                currentStatistics.append(calculateAngularStatistics(subTestGT, current));

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    currentIE[j, :] = [red, green, blue];

                currentResults.append(currentIE);

        best = 0;
        for i in range(1, trialsCount):
            if (currentStatistics[i][1] < currentStatistics[best][1]):
                best = i;

        ie[start:start + len(testGT), :] = currentResults[best];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test16(mainOutputPath, resultsPrefix, tasksN, taskPrefix, taskSuffix, lowerW, upperW, lowerB, upperB, lowerOrdinal,
           upperOrdinal):
    ns = [30];
    es = [10, 20, 50, 100];
    learningRates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05];
    bss = [32, 64, 128, 256];
    ds = ["lgb"];
    confs = ["s2_s2_s2_s1"];
    # confs=["s8_s1", "s16_s1", "s4_s4_s1", "s2_s2_s2_s1", "s2_s2_s1", "s4_s1", "s16_s4_s1"];
    suffixes = range(lowerOrdinal, upperOrdinal + 1);

    tasks = [];
    for dataset in ds:
        for n in ns:
            for epochs in es:
                for batchSize in bss:
                    for learningRate in learningRates:
                        for configuration in confs:
                            for suffix in suffixes:
                                outputPath = resultsPrefix + dataset + "_" + str(n) + "_" + str(epochs) + "_" + str(
                                    batchSize) + "_" + str(learningRate) + "_" + configuration + "_" + str(
                                    suffix) + ".txt";
                                tasks.append("python task.py " + dataset + " " + str(n) + " " + str(epochs) + " " + str(
                                    batchSize) + " " + str(
                                    learningRate) + " " + configuration + " " + outputPath + " " + str(
                                    lowerW) + " " + str(upperW) + " " + str(lowerB) + " " + str(upperB));

    separateTasks = [];
    for i in range(tasksN):
        separateTasks.append([]);

    current = 0;
    for i in range(len(tasks)):
        separateTasks[i % tasksN].append(tasks[i]);

    for i in range(tasksN):
        with open(taskPrefix + str(i) + taskSuffix, "w") as f:
            for task in separateTasks[i]:
                f.write(task + "\n");

    with open(mainOutputPath, "w") as f:
        for i in range(tasksN):
            f.write(taskPrefix + str(i) + taskSuffix + " &\n");


def Test17(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration);

    gt = loadPlainMatrix(prefix + ".txt");

    featuresRPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    featuresBPath = prefix + "_b_test_nd" + str(inputN) + ".txt";
    features = np.concatenate((loadPlainMatrix(featuresRPath), loadPlainMatrix(featuresBPath)), axis=1);

    inputN *= 2;

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], -1, 1));
            bias = tf.Variable(tf.random_uniform([n, 1], -1, 1));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test18(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", trials=1,
           outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration);

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], -1, 1));
            bias = tf.Variable(tf.random_uniform([n, 1], -1, 1));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            cie = np.copy(trainGT);
            for j in range(len(trainGT)):
                red = session.run(re, feed_dict={x: trainFeatures[j:j + 1, :].transpose(), r: [trainGT[j, 0:1]],
                                                 g: [trainGT[j, 1:2]], b: [trainGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                cie[j, :] = [red, green, blue];
            cs = calculateAngularStatistics(trainGT, cie);
            print(cs);

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test19(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getSubTrainFold(gt, i);
        trainFeatures = getSubTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);
        subTestGT = getSubTestFold(gt, i);
        subTestFeatures = getSubTestFold(features, i);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;
            be = p[0] * re + p[1];
            ge = 1 - re - be;

            # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
            # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];
            currentResults = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(subTestGT);
                currentIE = np.copy(testGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model" + str(trial));

                    for i in range(epochs):
                        featureBatches = getBatches(trainFeatures, batchSize);
                        dataBatches = getBatches(trainGT, batchSize);
                        for j in range(len(featureBatches)):
                            session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                          g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                    for j in range(len(subTestGT)):
                        red = session.run(re,
                                          feed_dict={x: subTestFeatures[j:j + 1, :].transpose(), r: [subTestGT[j, 0:1]],
                                                     g: [subTestGT[j, 1:2]], b: [subTestGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        current[j, :] = [red, green, blue];

                    currentStatistics.append(calculateAngularStatistics(subTestGT, current));

                    for j in range(len(testGT)):
                        red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                         g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        currentIE[j, :] = [red, green, blue];

                    currentResults.append(currentIE);

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                saver.restore(session, "./model" + str(best));

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test20(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

            re = finalOutput;
            be = p[0] * re + p[1];
            ge = 1 - re - be;

            # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
            # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];
            currentResults = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);
                currentIE = np.copy(testGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                    for i in range(epochs):
                        featureBatches = getBatches(trainFeatures, batchSize);
                        dataBatches = getBatches(trainGT, batchSize);
                        for j in range(len(featureBatches)):
                            session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                          g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                    for j in range(len(trainGT)):
                        red = session.run(re, feed_dict={x: trainFeatures[j:j + 1, :].transpose(), r: [trainGT[j, 0:1]],
                                                         g: [trainGT[j, 1:2]], b: [trainGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        current[j, :] = [red, green, blue];

                    currentStatistics.append(calculateAngularStatistics(trainGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test21(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        subTrainList = trainList[:];
        subTestIdx = subTrainList[-1];
        subTrainList.remove(subTestIdx);

        subTrainGT = getFolds(gt, subTrainList);
        subTrainFeatures = getFolds(features, subTrainList);

        subTestGT = getFolds(gt, [subTestIdx]);
        subTestFeatures = getFolds(features, [subTestIdx]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

            re = finalOutput;
            be = p[0] * re + p[1];
            ge = 1 - re - be;

            # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
            # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];
            currentResults = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(subTestGT);
                currentIE = np.copy(testGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                    for i in range(epochs):
                        featureBatches = getBatches(subTrainFeatures, batchSize);
                        dataBatches = getBatches(subTrainGT, batchSize);
                        for j in range(len(featureBatches)):
                            session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                          g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                    for j in range(len(subTestGT)):
                        red = session.run(re,
                                          feed_dict={x: subTestFeatures[j:j + 1, :].transpose(), r: [subTestGT[j, 0:1]],
                                                     g: [subTestGT[j, 1:2]], b: [subTestGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        current[j, :] = [red, green, blue];

                    currentStatistics.append(calculateAngularStatistics(subTestGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test22(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]");

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = list(range(15));
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;

            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:];
                    subTestIdx = subTrainList[ii];
                    subTrainList.remove(subTestIdx);

                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);

                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);

                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentStatistics.append(calculateAngularStatistics(trainGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                # os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test23(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        subTrainList = trainList[:];
        subTestIdx = subTrainList[-1];
        subTrainList.remove(subTestIdx);

        subTrainGT = getFolds(gt, subTrainList);
        subTrainFeatures = getFolds(features, subTrainList);

        subTestGT = getFolds(gt, [subTestIdx]);
        subTestFeatures = getFolds(features, [subTestIdx]);

        subTrainGT = np.copy(trainGT);
        subTrainFeatures = np.copy(trainFeatures);

        subTestGT = np.copy(testGT);
        subTestFeatures = np.copy(testFeatures);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

            re = finalOutput;
            be = p[0] * re + p[1];
            ge = 1 - re - be;

            # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
            # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];
            currentResults = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(subTestGT);
                currentIE = np.copy(testGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                    for i in range(epochs):
                        featureBatches = getBatches(subTrainFeatures, batchSize);
                        dataBatches = getBatches(subTrainGT, batchSize);
                        for j in range(len(featureBatches)):
                            session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                          g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                    for j in range(len(subTestGT)):
                        red = session.run(re,
                                          feed_dict={x: subTestFeatures[j:j + 1, :].transpose(), r: [subTestGT[j, 0:1]],
                                                     g: [subTestGT[j, 1:2]], b: [subTestGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        current[j, :] = [red, green, blue];

                    currentStatistics.append(calculateAngularStatistics(subTestGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test24(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;

            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:];
                    subTestIdx = subTrainList[ii];
                    subTrainList.remove(subTestIdx);

                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);

                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);

                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        errors = [];
                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                            currentResult = np.copy(subTestGT);
                            for j in range(len(subTestGT)):
                                red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                                 r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                                 b: [subTestGT[j, 2:3]]});
                                blue = p[0] * red + p[1];
                                green = 1 - red - blue;
                                currentResult[j, :] = [red, green, blue];
                            errors.append(calculateAngularStatistics(currentResult, subTestGT)[1]);

                        minIdx = 0;
                        for i in range(len(errors)):
                            if (errors[i] < errors[minIdx]):
                                minIdx = i;

                        print(errors);
                        print("minIdx:", minIdx);

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentStatistics.append(calculateAngularStatistics(trainGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                for i in range(epochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test25(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;

            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:];
                    subTestIdx = subTrainList[ii];
                    subTrainList.remove(subTestIdx);

                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);

                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);

                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentStatistics.append(calculateAngularStatistics(trainGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                subTrainList = trainList[:];
                subTestIdx = subTrainList[-1];
                subTrainList.remove(subTestIdx);

                subTrainGT = getFolds(gt, subTrainList);
                subTrainFeatures = getFolds(features, subTrainList);

                subTestGT = getFolds(gt, [subTestIdx]);
                subTestFeatures = getFolds(features, [subTestIdx]);

                p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                for i in range(epochs):
                    featureBatches = getBatches(subTrainFeatures, batchSize);
                    dataBatches = getBatches(subTrainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                    part = np.copy(testGT);
                    for j in range(len(testGT)):
                        red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                         g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        part[j, :] = [red, green, blue];
                    subPart = np.copy(subTestGT);
                    for j in range(len(subTestGT)):
                        red = session.run(re,
                                          feed_dict={x: subTestFeatures[j:j + 1, :].transpose(), r: [subTestGT[j, 0:1]],
                                                     g: [subTestGT[j, 1:2]], b: [subTestGT[j, 2:3]]});
                        blue = p[0] * red + p[1];
                        green = 1 - red - blue;
                        subPart[j, :] = [red, green, blue];
                    print(calculateAngularStatistics(testGT, part)[1],
                          calculateAngularStatistics(subTestGT, subPart)[1]);

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test26(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
            bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;

        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        currentStatistics = [];

        saver = tf.train.Saver(max_to_keep=1000);

        with tf.Session() as session:
            session.run(init);

            saver.save(session, "./model_" + tag);

            p = np.polyfit(gt[:, 0], gt[:, 2], 1);
            for trial in range(10):
                # saver.restore(session, "./model_"+tag);
                session.run(init);

                for i in range(epochs):
                    featureBatches = getBatches(features, batchSize);
                    dataBatches = getBatches(gt, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                current = np.copy(gt);
                for j in range(len(gt)):
                    red = session.run(re,
                                      feed_dict={x: features[j:j + 1, :].transpose(), r: [gt[j, 0:1]], g: [gt[j, 1:2]],
                                                 b: [gt[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    current[j, :] = [red, green, blue];

                print(calculateAngularStatistics(gt, current)[1]);


def Test27(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, testIdx=0, subTestIdx1=14, subTestIdx2=13):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    trainList = range(15);
    trainList.remove(testIdx);

    trainGT = getFolds(gt, trainList);
    trainFeatures = getFolds(features, trainList);

    testGT = getFolds(gt, [testIdx]);
    testFeatures = getFolds(features, [testIdx]);

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
            bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;

        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        currentStatistics = [];

        current = np.copy(trainGT);

        with tf.Session() as session:
            session.run(init);

            subStart = 0;

            p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

            subTrainList = trainList[:];
            subTrainList.remove(subTestIdx1);
            subTrainList.remove(subTestIdx2);

            subTrainGT = getFolds(gt, subTrainList);
            subTrainFeatures = getFolds(features, subTrainList);

            subTestGT1 = getFolds(gt, [subTestIdx1]);
            subTestFeatures1 = getFolds(features, [subTestIdx1]);

            subTestGT2 = getFolds(gt, [subTestIdx2]);
            subTestFeatures2 = getFolds(features, [subTestIdx2]);

            p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

            for i in range(epochs):
                featureBatches = getBatches(subTrainFeatures, batchSize);
                dataBatches = getBatches(subTrainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                part = np.copy(testGT);
                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    part[j, :] = [red, green, blue];
                subPart1 = np.copy(subTestGT1);
                for j in range(len(subTestGT1)):
                    red = session.run(re,
                                      feed_dict={x: subTestFeatures1[j:j + 1, :].transpose(), r: [subTestGT1[j, 0:1]],
                                                 g: [subTestGT1[j, 1:2]], b: [subTestGT1[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    subPart1[j, :] = [red, green, blue];
                subPart2 = np.copy(subTestGT2);
                for j in range(len(subTestGT2)):
                    red = session.run(re,
                                      feed_dict={x: subTestFeatures2[j:j + 1, :].transpose(), r: [subTestGT2[j, 0:1]],
                                                 g: [subTestGT2[j, 1:2]], b: [subTestGT2[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    subPart2[j, :] = [red, green, blue];
                print(calculateAngularStatistics(testGT, part)[1], calculateAngularStatistics(subTestGT1, subPart1)[1],
                      calculateAngularStatistics(subTestGT2, subPart2)[1]);


def Test28(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, testIdx=0, subTestIdx1=14, subTestIdx2=13):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    trainList = range(15);
    trainList.remove(testIdx);

    trainGT = getFolds(gt, trainList);
    trainFeatures = getFolds(features, trainList);

    testGT = getFolds(gt, [testIdx]);
    testFeatures = getFolds(features, [testIdx]);

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);
        p0 = tf.placeholder(tf.float32, shape=[1, None]);
        p1 = tf.placeholder(tf.float32, shape=[1, None]);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
            bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p0 * re + p1;
        ge = 1 - re - be;

        loss = 1 - (r * re + g * ge + b * be) / tf.sqrt((re * re + ge * ge + be * be) * (r * r + g * g + b * b));
        # loss=tf.square(r-re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        currentStatistics = [];

        current = np.copy(trainGT);

        with tf.Session() as session:
            session.run(init);

            subStart = 0;

            subTrainList = trainList[:];
            subTrainList.remove(subTestIdx1);
            subTrainList.remove(subTestIdx2);

            subTrainGT = getFolds(gt, subTrainList);
            subTrainFeatures = getFolds(features, subTrainList);

            subTestGT1 = getFolds(gt, [subTestIdx1]);
            subTestFeatures1 = getFolds(features, [subTestIdx1]);

            subTestGT2 = getFolds(gt, [subTestIdx2]);
            subTestFeatures2 = getFolds(features, [subTestIdx2]);

            p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

            for i in range(epochs):
                featureBatches = getBatches(subTrainFeatures, batchSize);
                dataBatches = getBatches(subTrainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]],
                                                  p0: p[0] * np.ones((1, len(dataBatches[j]))),
                                                  p1: p[1] * np.ones((1, len(dataBatches[j])))});

                part = np.copy(testGT);
                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    part[j, :] = [red, green, blue];
                subPart1 = np.copy(subTestGT1);
                for j in range(len(subTestGT1)):
                    red = session.run(re,
                                      feed_dict={x: subTestFeatures1[j:j + 1, :].transpose(), r: [subTestGT1[j, 0:1]],
                                                 g: [subTestGT1[j, 1:2]], b: [subTestGT1[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    subPart1[j, :] = [red, green, blue];
                subPart2 = np.copy(subTestGT2);
                for j in range(len(subTestGT2)):
                    red = session.run(re,
                                      feed_dict={x: subTestFeatures2[j:j + 1, :].transpose(), r: [subTestGT2[j, 0:1]],
                                                 g: [subTestGT2[j, 1:2]], b: [subTestGT2[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    subPart2[j, :] = [red, green, blue];
                print(calculateAngularStatistics(testGT, part)[1], calculateAngularStatistics(subTestGT1, subPart1)[1],
                      calculateAngularStatistics(subTestGT2, subPart2)[1]);


def Test29(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration);

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], -1, 1));
            bias = tf.Variable(tf.random_uniform([n, 1], -1, 1));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p[0] * re + p[1];
        ge = 1 - re - be;

        # loss=tf.acos((r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b)));
        # loss=(r*re+g*ge+b*be)/tf.sqrt((re*re+ge*ge+be*be)*(r*r+g*g+b*b));
        loss = tf.square(r - re);
        train = tf.train.AdamOptimizer(learningRate).minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            train = tf.train.AdamOptimizer(learningRate).minimize(loss);

            init = tf.initialize_all_variables();
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test30(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration);

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));
        trainGT = getTrainFold(gt, i);
        trainFeatures = getTrainFold(features, i);

        testGT = getTestFold(gt, i);
        testFeatures = getTestFold(features, i);

        x = tf.placeholder(tf.float32, shape=[inputN, None]);
        r = tf.placeholder(tf.float32, shape=[1, None]);
        g = tf.placeholder(tf.float32, shape=[1, None]);
        b = tf.placeholder(tf.float32, shape=[1, None]);
        p0 = tf.placeholder(tf.float32, shape=[1, None]);
        p1 = tf.placeholder(tf.float32, shape=[1, None]);

        p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

        previous = x;
        previousN = inputN;
        for layerConfiguration in configuration.split("_"):
            t = layerConfiguration[0].lower();
            n = int(layerConfiguration[1:]);
            weights = tf.Variable(tf.random_uniform([n, previousN], -1, 1));
            bias = tf.Variable(tf.random_uniform([n, 1], -1, 1));
            current = tf.add(tf.matmul(weights, previous), bias);
            if (t == "l"):
                pass;
            elif (t == "s"):
                current = tf.sigmoid(current);
            elif (t == "t"):
                current = tf.tanh(current);
            elif (t == "r"):
                current = tf.nn.relu(current);
            previous = current;
            previousN = n;

        finalOutput = previous;

        re = finalOutput;
        be = p0 * re + p1;
        ge = 1 - re - be;

        loss = 1 - (r * re + g * ge + b * be) / tf.sqrt((re * re + ge * ge + be * be) * (r * r + g * g + b * b));
        # loss=tf.square(r-re);
        optimizer = tf.train.AdamOptimizer(learningRate);
        train = optimizer.minimize(loss);

        init = tf.initialize_all_variables();

        with tf.Session() as session:
            session.run(init);

            for i in range(epochs):
                featureBatches = getBatches(trainFeatures, batchSize);
                dataBatches = getBatches(trainGT, batchSize);
                for j in range(len(featureBatches)):
                    session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                  g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]],
                                                  p0: p[0] * np.ones((1, len(dataBatches[j]))),
                                                  p1: p[1] * np.ones((1, len(dataBatches[j])))});

            for j in range(len(testGT)):
                red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                 g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                blue = p[0] * red + p[1];
                green = 1 - red - blue;
                ie[start + j, :] = [red, green, blue];

        start += testGT.shape[0];

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test31(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;

            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;

                currentForEpochs = np.zeros((current.shape[0], current.shape[1], epochs));

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:];
                    subTestIdx = subTrainList[ii];
                    subTrainList.remove(subTestIdx);

                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);

                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);

                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                            for j in range(len(subTestGT)):
                                red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                                 r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                                 b: [subTestGT[j, 2:3]]});
                                blue = p[0] * red + p[1];
                                green = 1 - red - blue;
                                currentForEpochs[subStart + j, :, i] = [red, green, blue];

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentList = [];
                for ei in range(epochs):
                    currentList.append(calculateAngularStatistics(trainGT, currentForEpochs[:, :, ei])[1]);
                currentStatistics.append(currentList);

            best = 0;
            bestEpochs = 0;
            for i in range(1, trialsCount):
                for j in range(epochs):
                    if (currentStatistics[i][j] < currentStatistics[best][bestEpochs]):
                        best = i;
                        bestEpochs = j;
            bestEpochs += 1;

            with tf.Session() as session:

                p = np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                for i in range(bestEpochs):
                    featureBatches = getBatches(trainFeatures, batchSize);
                    dataBatches = getBatches(trainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test32(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]");

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = list(range(15));
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;

            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:];
                    subTestIdx = subTrainList[ii];
                    subTrainList.remove(subTestIdx);

                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);

                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);

                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentStatistics.append(calculateAngularStatistics(trainGT, current));

            best = 0;
            for i in range(1, trialsCount):
                if (currentStatistics[i][1] < currentStatistics[best][1]):
                    best = i;

            with tf.Session() as session:

                # p=np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                subTrainList = trainList[:];
                subTestIdx = subTrainList[-1];
                subTrainList.remove(subTestIdx);

                subTrainGT = getFolds(gt, subTrainList);
                subTrainFeatures = getFolds(features, subTrainList);

                subTestGT = getFolds(gt, [subTestIdx]);
                subTestFeatures = getFolds(features, [subTestIdx]);

                p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                for i in range(epochs):
                    featureBatches = getBatches(subTrainFeatures, batchSize);
                    dataBatches = getBatches(subTrainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                # os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test33(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration) + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]";

    gt = loadPlainMatrix(prefix + ".txt");

    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);

    ie = np.copy(gt);

    import random;
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        print("Fold " + str(i + 1));

        trainList = range(15);
        trainList.remove(i);

        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            for layerConfiguration in configuration.split("_"):
                t = layerConfiguration[0].lower();
                n = int(layerConfiguration[1:]);
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                current = tf.add(tf.matmul(weights, previous), bias);
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);
                previous = current;
                previousN = n;

            finalOutput = previous;

            re = finalOutput;

            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);

            init = tf.initialize_all_variables();

            currentStatistics = [];

            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);

                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;

                currentForEpochs = np.zeros((current.shape[0], current.shape[1], epochs));

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:];
                    subTestIdx = subTrainList[ii];
                    subTrainList.remove(subTestIdx);

                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);

                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);

                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                            for j in range(len(subTestGT)):
                                red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                                 r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                                 b: [subTestGT[j, 2:3]]});
                                blue = p[0] * red + p[1];
                                green = 1 - red - blue;
                                currentForEpochs[subStart + j, :, i] = [red, green, blue];

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentList = [];
                for ei in range(epochs):
                    currentList.append(calculateAngularStatistics(trainGT, currentForEpochs[:, :, ei])[1]);
                currentStatistics.append(currentList);

            best = 0;
            bestEpochs = 0;
            for i in range(1, trialsCount):
                for j in range(epochs):
                    if (currentStatistics[i][j] < currentStatistics[best][bestEpochs]):
                        best = i;
                        bestEpochs = j;
            bestEpochs += 1;

            with tf.Session() as session:

                # p=np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                subTrainList = trainList[:];
                subTestIdx = subTrainList[-1];
                subTrainList.remove(subTestIdx);

                subTrainGT = getFolds(gt, subTrainList);
                subTrainFeatures = getFolds(features, subTrainList);

                subTestGT = getFolds(gt, [subTestIdx]);
                subTestFeatures = getFolds(features, [subTestIdx]);

                p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                for i in range(epochs):
                    featureBatches = getBatches(subTrainFeatures, batchSize);
                    dataBatches = getBatches(subTrainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");


def Test35(prefix="gb", inputN=20, epochs=20, batchSize=64, learningRate=0.01, configuration="l1", lowerW=-1, upperW=1,
           lowerB=-1, upperB=1, trialsCount=1, outputPath=None):
    print("Dataset=" + prefix + ", input=" + str(inputN) + ", epochs=" + str(epochs) + ", batchSize=" + str(
        batchSize) + ", learningRate=" + str(learningRate) + ", configuration=" + configuration + ", w=[" + str(
        lowerW) + ", " + str(upperW) + "]" + ", b=[" + str(lowerB) + ", " + str(upperB) + "]");
    # ucita gt
    gt = loadPlainMatrix(prefix + ".txt");
    # ucita feature
    featuresPath = prefix + "_r_test_nd" + str(inputN) + ".txt";
    features = loadPlainMatrix(featuresPath);
    # kopira gt iz nekog razloga
    ie = np.copy(gt);

    import random;
    #tagovi za modele, bzvz
    tag = str(random.randint(0, 2 ** 64 - 1));

    import os;

    start = 0;
    for i in range(15):
        #dijeli na 15 foldova
        print("Fold " + str(i + 1));

        trainList = list(range(15));
        trainList.remove(i);

        #svi osim i-tog folda
        trainGT = getFolds(gt, trainList);
        trainFeatures = getFolds(features, trainList);

        # validacijski fold
        testGT = getFolds(gt, [i]);
        testFeatures = getFolds(features, [i]);

        with tf.Graph().as_default():
            # input
            x = tf.placeholder(tf.float32, shape=[inputN, None]);
            # rgb
            r = tf.placeholder(tf.float32, shape=[1, None]);
            g = tf.placeholder(tf.float32, shape=[1, None]);
            b = tf.placeholder(tf.float32, shape=[1, None]);

            previous = x;
            previousN = inputN;
            # radi layere prema konfiguraciji
            for layerConfiguration in configuration.split("_"):
                #aktivacijska fja
                t = layerConfiguration[0].lower();
                #broj neurona
                n = int(layerConfiguration[1:]);
                # inicijalizira wetghtove i bias
                weights = tf.Variable(tf.random_uniform([n, previousN], lowerW, upperW));
                bias = tf.Variable(tf.random_uniform([n, 1], lowerB, upperB));
                # izlaz iz sloja
                current = tf.add(tf.matmul(weights, previous), bias);

                #assigna aktivacijsku
                if (t == "l"):
                    pass;
                elif (t == "s"):
                    current = tf.sigmoid(current);
                elif (t == "t"):
                    current = tf.tanh(current);
                elif (t == "r"):
                    current = tf.nn.relu(current);

                #spaja slojeve
                previous = current;
                previousN = n;

            # izlazni sloj
            finalOutput = previous;
            re = finalOutput;

            #loss fja
            loss = tf.square(r - re);
            optimizer = tf.train.AdamOptimizer(learningRate);
            train = optimizer.minimize(loss);
            init = tf.initialize_all_variables();

            currentStatistics = [];
            saver = tf.train.Saver(max_to_keep=1000);

            for trial in range(trialsCount):
                print("\tTrial " + str(trial + 1));
                current = np.copy(trainGT);

                with tf.Session() as session:
                    session.run(init);
                    saver.save(session, "./model_" + tag + "_" + str(trial));

                subStart = 0;
                currentForEpochs = np.zeros((current.shape[0], current.shape[1], epochs));

                for ii in range(14):

                    # print("\t\tSubfold "+str(ii+1));

                    subTrainList = trainList[:]; #0 - 15 indexi
                    subTestIdx = subTrainList[ii]; #validacijski fold
                    subTrainList.remove(subTestIdx);

                    # subfold za treniranje
                    subTrainGT = getFolds(gt, subTrainList);
                    subTrainFeatures = getFolds(features, subTrainList);
                    # subfold validacijski
                    subTestGT = getFolds(gt, [subTestIdx]);
                    subTestFeatures = getFolds(features, [subTestIdx]);

                    p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                    with tf.Session() as session:
                        session.run(init);
                        # restora model i pocinje trenirat
                        saver.restore(session, "./model_" + tag + "_" + str(trial));

                        for i in range(epochs):
                            featureBatches = getBatches(subTrainFeatures, batchSize);
                            dataBatches = getBatches(subTrainGT, batchSize);
                            for j in range(len(featureBatches)):
                                session.run(train,
                                            feed_dict={x: featureBatches[j].transpose(),
                                                       r: [dataBatches[j][:, 0]],
                                                       g: [dataBatches[j][:, 1]],
                                                       b: [dataBatches[j][:, 2]]});

                            for j in range(len(subTestGT)):
                                red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                                 r: [subTestGT[j, 0:1]],
                                                                 g: [subTestGT[j, 1:2]],
                                                                 b: [subTestGT[j, 2:3]]});
                                blue = p[0] * red + p[1];
                                green = 1 - red - blue;
                                currentForEpochs[subStart + j, :, i] = [red, green, blue];

                        for j in range(len(subTestGT)):
                            red = session.run(re, feed_dict={x: subTestFeatures[j:j + 1, :].transpose(),
                                                             r: [subTestGT[j, 0:1]], g: [subTestGT[j, 1:2]],
                                                             b: [subTestGT[j, 2:3]]});
                            blue = p[0] * red + p[1];
                            green = 1 - red - blue;
                            current[subStart + j, :] = [red, green, blue];

                    subStart += subTestGT.shape[0];

                currentList = [];
                for ei in range(epochs):
                    currentList.append(calculateAngularStatistics(trainGT, currentForEpochs[:, :, ei])[1]);
                currentStatistics.append(currentList);

            best = 0;
            bestEpochs = 0;
            for i in range(1, trialsCount):
                for j in range(epochs):
                    if (currentStatistics[i][j] < currentStatistics[best][bestEpochs]):
                        best = i;
                        bestEpochs = j;
            bestEpochs += 1;

            with tf.Session() as session:

                # p=np.polyfit(trainGT[:, 0], trainGT[:, 2], 1);

                saver.restore(session, "./model_" + tag + "_" + str(best));

                subTrainList = trainList[:];
                subTestIdx = subTrainList[-1];
                subTrainList.remove(subTestIdx);

                subTrainGT = getFolds(gt, subTrainList);
                subTrainFeatures = getFolds(features, subTrainList);

                subTestGT = getFolds(gt, [subTestIdx]);
                subTestFeatures = getFolds(features, [subTestIdx]);

                p = np.polyfit(subTrainGT[:, 0], subTrainGT[:, 2], 1);

                for i in range(epochs):
                    featureBatches = getBatches(subTrainFeatures, batchSize);
                    dataBatches = getBatches(subTrainGT, batchSize);
                    for j in range(len(featureBatches)):
                        session.run(train, feed_dict={x: featureBatches[j].transpose(), r: [dataBatches[j][:, 0]],
                                                      g: [dataBatches[j][:, 1]], b: [dataBatches[j][:, 2]]});

                for j in range(len(testGT)):
                    red = session.run(re, feed_dict={x: testFeatures[j:j + 1, :].transpose(), r: [testGT[j, 0:1]],
                                                     g: [testGT[j, 1:2]], b: [testGT[j, 2:3]]});
                    blue = p[0] * red + p[1];
                    green = 1 - red - blue;
                    ie[start + j, :] = [red, green, blue];

            start += testGT.shape[0];

            for trial in range(trialsCount):
                # os.remove("./model_" + tag + "_" + str(trial));
                pass;

    statistics = calculateAngularStatistics(gt, ie);
    print(statistics);

    if (outputPath is not None):
        with open(outputPath, "w") as f:
            for row in ie:
                f.write(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + "\n");

def main():
    # Test1();

    # Test2();

    # Test3();

    # Test4();

    # Test5("gb", 30, 1000, 128, 0.001);

    # Test6("gb", 30, 1000, 64, 0.001);

    # Test7("gb", 50, 100, 64, 0.01, "s8_s4_s2_l1", "tmp.txt");

    # Test8();

    # Test9("tasks.sh", "results/", 5, "./task_", ".sh");

    # Test7("lgb", 30, 10, 64, 0.01, "s8_s1");

    # Test10("lgb", 30, 200, 64, 0.001, "s2_s2_s2_s2_s1", 0, 0, 0, 0);

    # Test11("tasks.sh", "results0/", 3, "./task_", ".sh", 0, 0);

    # Test13("results0/lgb*.txt", "lgb");

    # Test13("results0/gb_30_10_64_0.01_s8_s1.txt", "gb");

    # Test14("lgb", 30, 50, 64, 0.01, "s8_s1", -1, 1, -1, 1, 10, None);

    # Test15("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 3, None);

    # Test16("tasks.sh", "results_r/", 3, "./task_", ".sh", -1, 1, -1, 1, 11, 20);

    # Test13("results_r/lgb*s2_s2_s2_s1*.txt", "lgb");

    # smaller max. error
    # Test7("lgb", 30, 10, 16, 0.01, "s4_s4_s1");

    # Test7("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s2_s1");

    # OK
    # Test7("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1");

    # Test17("lgb", 10, 30, 64, 0.01, "s2_s2_s2_s1");

    # Test15("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # Test18("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1");

    # something seems to be wrong here...
    # Test19("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # Test20("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # Test21("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # Test22("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # super
    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);

    # Test23("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 1);

    # Test24("lgb", 30, 20, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);

    # Test25("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 1);

    # Test26("lgb", 30, 10, 64, 0.01, "s2_s2_s2_s1", -1, 1, -1, 1, 5);

    # Test27("lgb", 30, 500, 16, 0.001, "s8_s1", -1, 1, -1, 1, 0, 14, 13);

    # Test28("lgb", 30, 500, 16, 0.001, "s8_s1", -1, 1, -1, 1, 0, 14, 13);

    # Test29("lgb", 30, 10, 64, 0.01, "s8_s1");

    # Test7("lgb", 30, 10, 64, 0.01, "s8_s1");

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);

    # Test30("gb", 30, 10, 64, 0.01, "s8_s1");

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 25);

    # Test31("lgb", 30, 50, 64, 0.001, "s8_s1", -1, 1, -1, 1, 5);

    # Test22("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);

    # same as Test22, but with a reduced training set after the selection of the best initial values
    # Test32("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);

    # Test32("lgb", 30, 10, 64, 0.005, "s8_s1", -1, 1, -1, 1, 3);

    # same as Test31, but with a reduced training set after the selection of the best initial values
    # Test33("lgb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);

    # Test32("gb", 30, 10, 64, 0.01, "s8_s1", -1, 1, -1, 1, 5);
    # Test32("gb", 30, 10, 64, 0.01, "r8_r5_s1", -1, 1, -1, 1, 5);

    # Test32("lgb", 30, 10, 64, 0.012, "s8_s1", -1, 1, -1, 1, 5);

    # Test35("lgb", 30, 10, 64, 0.012, "r8_r8_s1", -1, 1, -1, 1, 5);
    Test35("lgb", 20, 10, 64, 0.012, "r5_r5_s1", -1, 1, -1, 1, 5);

    pass;


if (__name__ == "__main__"):
    main();
