import pandas as pd
import numpy as np
import os
import argparse
import subprocess
import time
import concurrent.futures
import traceback

import datetime
try:
    import cudf
    import rmm
    import cuspatial
    import cuml
    from numba import cuda
except ModuleNotFoundError:
    pass

class Location:
    def __init__(self, minX, minY, width, height):
        self.minX = minX
        self.minY = minY
        self.width = width
        self.height = height
        self.maxX = minX + width
        self.maxY = minY + height

class RideshareData:
    def __init__(
        self,
        driverCount,
        riderCount,
        rideCount,
        locationCount,
        rideReqCount,
        driverStatusCount,
    ):

        self.driverCount = driverCount
        self.riderCount = riderCount
        self.rideCount = rideCount
        self.locationCount = locationCount
        self.rideReqCount = rideReqCount
        self.driverStatusCount = driverStatusCount

    def _genLocations(self, outputPath):
        self.locations = []

        grids = int(np.floor(np.sqrt(self.locationCount)))

        width = 1.0 / grids
        height = 1.0 / grids
        for i in range(self.locationCount):
            x = int(i % grids)
            y = int(i / grids)

            l = Location(x * width, y * height, width, height)
            self.locations.append(l)

        locationId = np.arange(self.locationCount)

        minX = np.zeros(self.locationCount)
        minY = np.zeros(self.locationCount)
        maxX = np.zeros(self.locationCount)
        maxY = np.zeros(self.locationCount)
        for i in range(self.locationCount):
            minX[i] = self.locations[i].minX
            minY[i] = self.locations[i].minY
            maxX[i] = self.locations[i].maxX
            maxY[i] = self.locations[i].maxY

        locationDict = {
            'locationId': locationId,
            'boundsMinX': minX,
            'boundsMinY': minY,
            'boundsMaxX': maxX,
            'boundsMaxY': maxY,
        }
        self.locationTable = pd.DataFrame(data=locationDict)

        featureCount = 10
        for i in range(0, featureCount):
            self.locationTable['c{}'.format(i)] = np.random.rand(self.locationCount)

        header = ['locationId', 'boundsMinX', 'boundsMinY', 'boundsMaxX', 'boundsMaxY']
        for i in range(0, featureCount):
            header.append('c{}'.format(i))
        self.locationTable.to_csv(
            os.path.join(outputPath, 'location.csv'),
            index = False,
            columns = header
        )

    def _genRideReqTable(self, outputPath):
        locationIndex = np.random.randint(len(self.locations), size=self.rideReqCount)

        rideId = np.arange(self.rideReqCount)

        startXLocation = np.random.rand(self.rideReqCount)
        startYLocation = np.random.rand(self.rideReqCount)

        startX = np.zeros(self.rideReqCount)
        startY = np.zeros(self.rideReqCount)
        for i in range(self.rideReqCount):
            loc = self.locations[locationIndex[i]]
            startX[i] = loc.minX + loc.width * startXLocation[i]
            startY[i] = loc.minY + loc.height * startYLocation[i]

        seatCount = np.random.randint(4, size=self.rideReqCount)

        riderId = np.random.randint(self.riderCount, size=self.rideReqCount)
        time = np.arange(self.rideReqCount)

        bins = np.arange(0, self.rideReqCount, 10000)
        timeBins = np.digitize(time, bins)

        rideReqDict = {
            'time': time,
            'timeBins': timeBins,
            'rideId': rideId,
            'riderId': riderId,
            'seatCount': seatCount,
            'startX': startX,
            'startY': startY,
        }
        self.rideReqTable = pd.DataFrame(data=rideReqDict)
        self.rideReqTable['timeDate'] = pd.to_datetime(self.rideReqTable['time'], unit='s')
        self.rideReqTable = self.rideReqTable.sort_values(by=['time'])

        featureCount = 12
        for i in range(0, featureCount):
            self.rideReqTable['c{}'.format(i)] = np.random.rand(self.rideReqCount)

        header = ['time', 'timeBins', 'timeDate', 'rideId', 'riderId', 'seatCount', 'startX', 'startY']
        for i in range(0, featureCount):
            header.append('c{}'.format(i))
        self.rideReqTable.to_csv(
            os.path.join(outputPath, 'ride_req.csv'),
            index = False,
            columns = header
        )

    def _genDriverStatusTable(self, outputPath):
        posX = np.random.rand(self.driverStatusCount)
        posY = np.random.rand(self.driverStatusCount)

        time = np.arange(self.driverStatusCount)
        driverId = np.random.randint(self.driverCount, size=self.driverStatusCount)

        driverDict = {
            'time': time,
            'driverId': driverId,
            'posX': posX,
            'posY': posY,
        }
        self.driverStatusTable = pd.DataFrame(data=driverDict)

        featureCount = 12
        for i in range(0, featureCount):
            self.driverStatusTable['c{}'.format(i)] = np.random.rand(self.driverStatusCount)

        self.driverStatusTable['timeDate'] = pd.to_datetime(self.driverStatusTable['time'], unit='s')
        self.driverStatusTable = self.driverStatusTable.sort_values(by=['time'])

        header = ['time', 'timeDate', 'driverId', 'posX', 'posY']
        for i in range(0, featureCount):
            header.append('c{}'.format(i))
        self.driverStatusTable.to_csv(
            os.path.join(outputPath, 'driver_status.csv'),
            index = False,
            columns = header
        )

    def _genDriverTable(self, outputPath):
        driverId = np.arange(self.driverCount)
        seatCount = np.random.randint(4, size=self.driverCount)
        rating = np.random.rand(self.driverCount)

        driverDict = {
            'driverId': driverId,
            'seatCount': seatCount,
            'rating': rating,
        }
        self.driverTable = pd.DataFrame(data=driverDict)

        header = ['driverId', 'seatCount', 'rating']
        self.driverTable.to_csv(
            os.path.join(outputPath, 'driver.csv'),
            index = False,
            columns = header
        )

    def _genRiderTable(self, outputPath):
        riderId = np.arange(self.riderCount)
        rating = np.random.rand(self.riderCount)

        time = np.arange(self.riderCount)

        riderDict = {
            'riderId': riderId,
            'rating': rating,
            'signupTime': time,
        }
        self.riderTable = pd.DataFrame(data=riderDict)

        self.riderTable['signupTimeDate'] = pd.to_datetime(self.riderTable['signupTime'], unit='s')

        header = ['riderId', 'rating', 'signupTime', 'signupTimeDate']
        self.riderTable.to_csv(
            os.path.join(outputPath, 'rider.csv'),
            index = False,
            columns = header
        )

    def _genRideTable(self, outputPath):
        rideId = np.arange(self.rideCount)

        riderId = np.random.randint(self.riderCount, size=self.rideCount)
        driverId = np.random.randint(self.driverCount, size=self.rideCount)

        startX = np.random.rand(self.rideCount)
        startY = np.random.rand(self.rideCount)
        endX = np.random.rand(self.rideCount)
        endY = np.random.rand(self.rideCount)

        startTime = np.random.randint(self.riderCount, size=self.rideCount)

        rideDict = {
            'rideId': rideId,
            'riderId': riderId,
            'driverId': driverId,
            'startX': startX,
            'startY': startY,
            'endX': endX,
            'endY': endY,
            'startTime': startTime,
        }
        self.rideTable = pd.DataFrame(data=rideDict)

        self.rideTable['startTimeDate'] = pd.to_datetime(self.rideTable['startTime'], unit='s')

        featureCount = 12
        for i in range(0, featureCount):
            self.rideTable['c{}'.format(i)] = np.random.rand(self.rideCount)

        predCount = 5
        for i in range(0, predCount):
            self.rideTable['pred{}'.format(i)] = np.random.rand(self.rideCount)

        header = ['rideId', 'riderId', 'driverId', 'startX', 'startY', 'endX', 'endY', 'startTime', 'startTimeDate']
        for i in range(0, featureCount):
            header.append('c{}'.format(i))
        for i in range(0, predCount):
            header.append('pred{}'.format(i))
        self.rideTable.to_csv(
            os.path.join(outputPath, 'ride.csv'),
            index = False,
            columns = header
        )

    def gen(self, outputPath):
        print('Generating {}'.format(outputPath))
        try:
            os.makedirs(outputPath)
        except:
            pass

        self._genLocations(outputPath)
        self._genDriverTable(outputPath)
        self._genDriverStatusTable(outputPath)
        self._genRideReqTable(outputPath)
        self._genRiderTable(outputPath)
        self._genRideTable(outputPath)

        print('Done {}'.format(outputPath))

class QueryGen:
    def __init__(self, args):
        np.random.seed(0)
        self.args = args

    def genData(self):
        s = self.args.scale
        genAll = not self.args.queries

        rideshareQ1 = RideshareData(
            driverCount=s * 1000,
            riderCount=s * 10000,
            rideCount=0,
            locationCount=s * 1000,
            rideReqCount=s * 1000000,
            driverStatusCount=s * 10000,
        )
        rideshareQ2 = RideshareData(
            driverCount=s * 10000,
            riderCount=s * 10000,
            rideCount=0,
            locationCount=s * 1000,
            rideReqCount=s * 1000000,
            driverStatusCount=0,
        )
        rideshareQ3 = RideshareData(
            driverCount=s * 10000,
            riderCount=s * 1000,
            rideCount=0,
            locationCount=s * 100,
            rideReqCount=s * 1000000,
            driverStatusCount=s * 100000,
        )
        rideshareQ4 = RideshareData(
            driverCount=s * 100000,
            riderCount=s * 100000,
            rideCount=s * 100000,
            locationCount=s * 100,
            rideReqCount=s * 1000,
            driverStatusCount=0,
        )
        rideshareQ5 = RideshareData(
            driverCount=s * 1000,
            riderCount=0,
            rideCount=0,
            locationCount=s * 1000,
            rideReqCount=0,
            driverStatusCount=s * 100000,
        )
        rideshareQ6 = RideshareData(
            driverCount=s * 10000,
            riderCount=s * 10000,
            rideCount=0,
            locationCount=s * 100,
            rideReqCount=s * 100000,
            driverStatusCount=s * 10000,
        )
        rideshareQ7 = RideshareData(
            driverCount=s * 100000,
            riderCount=s * 100000,
            rideCount=s * 1000000,
            locationCount=s * 1000,
            rideReqCount=0,
            driverStatusCount=0,
        )
        rideshareQ8 = RideshareData(
            driverCount=s * 10000,
            riderCount=s * 10000,
            rideCount=s * 100000,
            locationCount=s * 100,
            rideReqCount=0,
            driverStatusCount=0,
        )
        rideshareQ9 = RideshareData(
            driverCount=s * 10000,
            riderCount=s * 10000,
            rideCount=0,
            locationCount=s * 1000,
            rideReqCount=s * 100000,
            driverStatusCount=s * 10000,
        )

        queries = {
            'query1': rideshareQ1,
            'query2': rideshareQ2,
            'query3': rideshareQ3,
            'query4': rideshareQ4,
            'query5': rideshareQ5,
            'query6': rideshareQ6,
            'query7': rideshareQ7,
            'query8': rideshareQ8,
            'query9': rideshareQ9,
        }

        queryFilter = dict(filter(lambda x: genAll or x[0] in self.args.queries, queries.items()))

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = { executor.submit(v.gen, 'data/s{}/{}'.format(s, k)): v for k, v in queryFilter.items()}
            for f in concurrent.futures.as_completed(futures):
                q = futures[f]
                try:
                    f.result()
                except Exception as e:
                    print(e)

class QueryRunGpu:
    def __init__(self, args):
        self.args = args

        if (self.args.debug):
            rmm.reinitialize(
                logging=True,
                log_file_name="memlog"
            )

    def _loadTables(
            self,
            queryPath
        ):

        scalePath = 'data/s{}'.format(self.args.scale)
        driverCsv = os.path.join(scalePath, queryPath, 'driver.csv')
        driverStatusCsv = os.path.join(scalePath, queryPath, 'driver_status.csv')
        rideReqCsv = os.path.join(scalePath, queryPath, 'ride_req.csv')
        rideCsv = os.path.join(scalePath, queryPath, 'ride.csv')
        riderCsv = os.path.join(scalePath, queryPath, 'rider.csv')
        locationCsv = os.path.join(scalePath, queryPath, 'location.csv')

        self.driverTable = cudf.read_csv(driverCsv)
        if self.driverTable.shape[0] > 0:
            self.driverTable.columns = ['drv.' + str(col) for col in self.driverTable]

        self.driverStatusTable = cudf.read_csv(driverStatusCsv)
        if self.driverStatusTable.shape[0] > 0:
            self.driverStatusTable.columns = ['drvStat.' + str(col) for col in self.driverStatusTable]

        self.rideReqTable = cudf.read_csv(rideReqCsv)
        if self.rideReqTable.shape[0] > 0:
            self.rideReqTable.columns = ['rideReq.' + str(col) for col in self.rideReqTable]

        self.rideTable = cudf.read_csv(rideCsv)
        if self.rideTable.shape[0] > 0:
            self.rideTable.columns = ['ride.' + str(col) for col in self.rideTable]

        self.riderTable = cudf.read_csv(riderCsv)
        if self.riderTable.shape[0] > 0:
            self.riderTable.columns = ['rider.' + str(col) for col in self.riderTable]

        self.locationTable = cudf.read_csv(locationCsv)
        if self.locationTable.shape[0] > 0:
            self.locationTable.columns = ['loc.' + str(col) for col in self.locationTable]

    def _createIndex(self, df, colName):
        out = cuspatial.quadtree_on_points(
            df[colName + 'X'],
            df[colName + 'Y'],
            0,
            1,
            0,
            1,
            1,
            15,
            20
        )

        return out

    def _polygonFromBox(
        self,
        box
    ):
        pointCount = box.shape[0] * 4
        polyOffset = cudf.Series(np.arange(box.shape[0]))
        ringOffset = cudf.Series(np.arange(start=0, step=4, stop=pointCount))

        xPoints = np.zeros(pointCount)
        yPoints = np.zeros(pointCount)

        xMin = box['xMin']
        xMax = box['xMax']
        yMin = box['yMin']
        yMax = box['yMax']

        xPoints[0::4] = xMin
        xPoints[1::4] = xMax
        xPoints[2::4] = xMax
        xPoints[3::4] = xMin

        yPoints[0::4] = yMax
        yPoints[1::4] = yMax
        yPoints[2::4] = yMin
        yPoints[3::4] = yMin

        return (polyOffset, ringOffset, xPoints, yPoints)

    def _createPoint(
        self,
        inDf,
        colName,
        radius,
    ):
        pandas = inDf.to_pandas()

        box = pd.DataFrame({
            'xMin': pandas[colName + 'X'] - radius,
            'xMax': pandas[colName + 'X'] + radius,
            'yMin': pandas[colName + 'Y'] - radius,
            'yMax': pandas[colName + 'Y'] + radius,
        })

        out = self._polygonFromBox(box)

        return out

    def _createBox(
        self,
        inDf,
        colName
    ):
        pandas = inDf.to_pandas()

        box = pd.DataFrame({
            'xMin': pandas[colName + 'MinX'],
            'xMax': pandas[colName + 'MaxX'],
            'yMin': pandas[colName + 'MinY'],
            'yMax': pandas[colName + 'MaxY'],
        })

        out = self._polygonFromBox(box)

        return out

    def _spatialJoinDist(
        self,
        ldf,
        rdf,
        lName,
        rName,
        lTree,
        polygon,
        dist
    ):
        (polyOffset, ringOffset, xPoints, yPoints) = polygon
        (points, tree) = lTree

        boundingBox = cuspatial.polygon_bounding_boxes(
            polyOffset,
            ringOffset,
            xPoints,
            yPoints
        )

        joinFilter = cuspatial.join_quadtree_and_bounding_boxes(
            tree,
            boundingBox,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            15
        )

        joinPolygon = cuspatial.quadtree_point_in_polygon(
            joinFilter,
            tree,
            points,
            ldf[lName + 'X'],
            ldf[lName + 'Y'],
            polyOffset,
            ringOffset,
            xPoints,
            yPoints,
        )

        # https://github.com/rapidsai/cuspatial/issues/284
        lGather = ldf.take(points.take(joinPolygon['point_index'])).reset_index(drop=True)
        rGather = rdf.take(joinPolygon['polygon_index']).reset_index(drop=True)

        dfConcat = cudf.concat([lGather, rGather], axis=1)
        dfConcat['distPred'] = False

        @cuda.jit
        def distPredFunc(lX, lY, rX, rY, out, dist):
            i = cuda.grid(1)
            if i < lX.shape[0]:
                dX = lX[i] - rX[i]
                dY = lY[i] - rY[i]
                dSquare = (dX * dX) + (dY * dY)
                out[i] = dSquare < (dist * dist)

        numbaTime = 0.0
        if dist > 0.0:
            startTime = time.time()
            distPredFunc.forall(dfConcat.shape[0])(
                dfConcat[lName + 'X'],
                dfConcat[lName + 'Y'],
                dfConcat[rName + 'X'],
                dfConcat[rName + 'Y'],
                dfConcat['distPred'],
                dist
            )
            endTime = time.time()
            numbaTime = endTime - startTime

            dfConcat = dfConcat[dfConcat['distPred']]

        return (dfConcat, numbaTime)

    def _spatialJoinKnn(
        self,
        ldf,
        rdf,
        lName,
        rName,
        k,
    ):
        assert(ldf.shape[0] == 1)

        @cuda.jit
        def distFunc(lX, lY, rX, rY, out):
            i = cuda.grid(1)
            if i < lX.shape[0]:
                dX = lX[i] - rX[i]
                dY = lY[i] - rY[i]
                dSquare = (dX * dX) + (dY * dY)
                out[i] = dSquare

        out = rdf
        out['dist'] = 0.0

        startTime = time.time()
        distFunc.forall(rdf.shape[0])(
            ldf[lName + 'X'],
            ldf[lName + 'Y'],
            rdf[rName + 'X'],
            rdf[rName + 'Y'],
            out['dist'],
        )
        endTime = time.time()
        numbaTime = endTime - startTime

        out = out.sort_values('dist')
        out = out.head(k)

        dummyR = ldf
        dummyR['dummy'] = 0
        out['dummy'] = 0

        out = out.merge(dummyR, on='dummy', how='outer')

        return (out, numbaTime)

    def _query1(self):
        self._loadTables('query1')

        self.driverStatusTable = self.driverStatusTable[self.driverStatusTable['drvStat.time'] < self.driverStatusTable.shape[0] / 500]
        self.rideReqTable = self.rideReqTable[self.rideReqTable['rideReq.time'] < self.rideReqTable.shape[0] / 10]

        radius = 0.025 / np.sqrt(self.args.scale)
        driverPolygon = self._createPoint(
            self.driverStatusTable,
            'drvStat.pos',
            radius
        )

        startTime = time.time()
        rideReqIndex = self._createIndex(
            self.rideReqTable,
            'rideReq.start',
        )

        (joinStatus, numbaTime) = self._spatialJoinDist(
            self.rideReqTable,
            self.driverStatusTable,
            'rideReq.start',
            'drvStat.pos',
            rideReqIndex,
            driverPolygon,
            radius
        )

        joinDriver = joinStatus.merge(self.driverTable, left_on='drvStat.driverId', right_on='drv.driverId')
        cond = joinDriver[joinDriver['rideReq.seatCount'] < joinDriver['drv.seatCount']]

        cond['count'] = 0
        group = cond[['drv.driverId', 'count']].groupby(
            ['drv.driverId'],
        ).agg({
            'count': 'count',
        }).reset_index()

        endTime = time.time()

        group.to_csv(
            'query1_gpu.csv',
            index=False
        )

        return endTime - startTime - numbaTime

    def _query2(self):
        self._loadTables('query2')

        rideReqIndex = self._createIndex(
            self.rideReqTable,
            'rideReq.start',
        )

        filterLocation = self.locationTable[self.locationTable['loc.locationId'] == 0]

        locationPolygon = self._createBox(
            filterLocation,
            'loc.bounds',
        )

        startTime = time.time()

        (joinLocation, numbaTime) = self._spatialJoinDist(
            self.rideReqTable,
            filterLocation,
            'rideReq.start',
            'loc.bounds',
            rideReqIndex,
            locationPolygon,
            0.0
        )

        joinLocation['count'] = 0
        group = joinLocation.groupby(
            ['rideReq.timeBins'],
        ).agg({
            'count': 'count',
        }).reset_index()

        out = group.sort_values('count')
        endTime = time.time()

        out.to_csv(
            'query2_gpu.csv',
            index=False,
        )

        return endTime - startTime - numbaTime

    def _query3(self):
        self._loadTables('query3')

        self.rideReqTable = self.rideReqTable[self.rideReqTable['rideReq.time'] < self.rideReqTable.shape[0] / 100]

        locationPolygon = self._createBox(
            self.locationTable,
            'loc.bounds',
        )

        startTime = time.time()

        rideReqIndex = self._createIndex(
            self.rideReqTable,
            'rideReq.start',
        )

        (joinLocation, numbaTime) = self._spatialJoinDist(
            self.rideReqTable,
            self.locationTable,
            'rideReq.start',
            'loc.bounds',
            rideReqIndex,
            locationPolygon,
            0.0
        )

        joinLocation['count'] = 0
        group = joinLocation.groupby(
            ['loc.locationId'],
        ).agg({
            'count': 'count',
        }).reset_index()
        out = group.sort_values('count')
        endTime = time.time()

        out.to_csv(
            'query3_gpu.csv',
            index=False,
        )
        return endTime - startTime - numbaTime

    def _query4(self):
        self._loadTables('query4')

        self.rideTable = self.rideTable[self.rideTable['ride.startTime'] < self.rideTable.shape[0] / 10]

        knn = cuml.neighbors.KNeighborsRegressor(n_neighbors=5)

        rideReqIndex = self._createIndex(
            self.rideReqTable,
            'rideReq.start',
        )

        locationFilter = self.locationTable[self.locationTable['loc.locationId'] == 0]
        locationPolygon = self._createBox(
            locationFilter,
            'loc.bounds',
        )

        startTime = time.time()

        rideIndex = self._createIndex(
            self.rideTable,
            'ride.start',
        )

        (joinRide, numbaTime0) = self._spatialJoinDist(
            self.rideTable,
            locationFilter,
            'ride.start',
            'loc.bounds',
            rideIndex,
            locationPolygon,
            0.0
        )

        featureNameRide = [
            'ride.c0',
            'ride.c1',
            'ride.c2',
            'ride.c3',
            'ride.c4',
            'ride.c5',
            'ride.c6',
            'ride.c7',
            'ride.c8',
            'ride.c9',
            'ride.c10',
            'ride.c11'
        ]
        featureNameRideReq = [
            'rideReq.c0',
            'rideReq.c1',
            'rideReq.c2',
            'rideReq.c3',
            'rideReq.c4',
            'rideReq.c5',
            'rideReq.c6',
            'rideReq.c7',
            'rideReq.c8',
            'rideReq.c9',
            'rideReq.c10',
            'rideReq.c11'
        ]

        trainX = joinRide[featureNameRide]
        trainY = joinRide['ride.pred0']
        knn.fit(trainX, trainY)

        (joinReq, numbaTime1) = self._spatialJoinDist(
            self.rideReqTable,
            locationFilter,
            'rideReq.start',
            'loc.bounds',
            rideReqIndex,
            locationPolygon,
            0.0
        )
        joinReq = joinReq.head(1)

        joinReq['predict'] = knn.predict(joinReq[featureNameRideReq])
        endTime = time.time()

        joinReq.to_csv(
            'query4_gpu.csv',
            index=False,
        )
        return endTime - startTime - numbaTime0 - numbaTime1

    def _query5(self):
        self._loadTables('query5')

        supportVec = 8
        xVal = np.random.rand(supportVec, 12)
        yVal = np.random.choice([-1.0, 1.0], size=supportVec)

        svm = cuml.SVC(kernel='poly', degree=2)
        svm.fit(xVal, yVal)

        featureName = [
            'drvStat.c0',
            'drvStat.c1',
            'drvStat.c2',
            'drvStat.c3',
            'drvStat.c4',
            'drvStat.c5',
            'drvStat.c6',
            'drvStat.c7',
            'drvStat.c8',
            'drvStat.c9',
            'drvStat.c10',
            'drvStat.c11'
        ]

        startTime = time.time()
        join = self.driverStatusTable.merge(self.driverTable, left_on='drvStat.driverId', right_on='drv.driverId')
        groupby = join.groupby(['drv.driverId'])[featureName].rolling(3, min_periods=1).mean()

        predict = svm.predict(groupby[featureName])
        endTime = time.time()

        groupby.to_csv(
            'query5_gpu.csv',
            index=False,
        )
        return endTime - startTime

    def _query6(self):
        self._loadTables('query6')

        self.rideReqTable = self.rideReqTable[self.rideReqTable['rideReq.time'] < self.rideReqTable.shape[0] / 10]

        rideReqIndex = self._createIndex(
            self.rideReqTable,
            'rideReq.start',
        )

        driverStatusIndex = self._createIndex(
            self.driverStatusTable,
            'drvStat.pos',
        )

        locationPolygon = self._createBox(
            self.locationTable,
            'loc.bounds',
        )

        trainX = {}
        for i in range(10):
            trainX['c{}'.format(i)] = np.random.rand(1000)
        trainX = cudf.DataFrame(trainX)

        trainY = np.random.choice([0.0, 1.0], size=1000)
        trainY = cudf.Series(trainY)

        linReg = cuml.LinearRegression()
        linReg.fit(trainX, trainY)

        startTime = time.time()

        (joinRideReq, numbaTime0) = self._spatialJoinDist(
            self.rideReqTable,
            self.locationTable,
            'rideReq.start',
            'loc.bounds',
            rideReqIndex,
            locationPolygon,
            0.0
        )

        joinRideReq['count'] = 0
        reqGroup = joinRideReq.groupby(
            ['loc.locationId'],
        ).agg({
            'count': 'count',
        }).reset_index()

        (joinDriver, numbaTime1) = self._spatialJoinDist(
            self.driverStatusTable,
            self.locationTable,
            'drvStat.pos',
            'loc.bounds',
            driverStatusIndex,
            locationPolygon,
            0.0
        )

        joinDriver['count'] = 0
        driverGroup = joinDriver.groupby(
            ['loc.locationId'],
        ).agg({
            'count': 'count',
        }).reset_index()

        join0 = reqGroup.merge(driverGroup, on='loc.locationId')
        join1 = join0.merge(self.locationTable, on='loc.locationId')

        featureName = [
            'loc.c0',
            'loc.c1',
            'loc.c2',
            'loc.c3',
            'loc.c4',
            'loc.c5',
            'loc.c6',
            'loc.c7',
            'loc.c8',
            'loc.c9',
        ]
        join1['infer'] = linReg.predict(join1[featureName])

        endTime = time.time()

        join1.to_csv(
            'query6_gpu.csv',
            index=False,
        )
        return endTime - startTime - numbaTime0 - numbaTime1

    def _query7(self):
        self._loadTables('query7')

        trainX = {}
        for i in range(12):
            trainX['c{}'.format(i)] = np.random.rand(1000)
        trainX = cudf.DataFrame(trainX)

        trainY = np.random.choice([0.0, 1.0], size=1000)
        trainY = cudf.Series(trainY)

        logReg = cuml.LogisticRegression()
        logReg.fit(trainX, trainY)

        self.riderTable = self.riderTable[self.riderTable['rider.signupTime'] < self.riderTable.shape[0] / 2]

        startTime = time.time()

        join0 = self.rideTable.merge(self.riderTable, left_on='ride.riderId', right_on='rider.riderId')
        join1 = join0.merge(self.driverTable, left_on='ride.driverId', right_on='drv.driverId')

        featureName = [
            'ride.c0',
            'ride.c1',
            'ride.c2',
            'ride.c3',
            'ride.c4',
            'ride.c5',
            'ride.c6',
            'ride.c7',
            'ride.c8',
            'ride.c9',
            'ride.c10',
            'ride.c11'
        ]

        groupRider = join1.groupby(
            ['ride.riderId'],
        )[featureName].mean()

        groupRider['predict'] = logReg.predict(groupRider)

        endTime = time.time()

        groupRider.to_csv(
            'query7_gpu.csv',
            index=False,
        )
        return endTime - startTime

    def _query8(self):
        self._loadTables('query8')

        train = {}
        for i in range(12):
            train['c{}'.format(i)] = np.random.rand(1000)
        train = cudf.DataFrame(train)
        kmeans = cuml.KMeans(n_clusters=8)
        kmeans.fit(train)

        rideIndex = self._createIndex(
            self.rideTable,
            'ride.start',
        )

        locationFilter = self.locationTable[self.locationTable['loc.locationId'] == 0]
        locationPolygon = self._createBox(
            locationFilter,
            'loc.bounds',
        )

        startTime = time.time()

        (joinRide, numbaTime) = self._spatialJoinDist(
            self.rideTable,
            locationFilter,
            'ride.start',
            'loc.bounds',
            rideIndex,
            locationPolygon,
            0.0
        )

        featureName = [
            'ride.c0',
            'ride.c1',
            'ride.c2',
            'ride.c3',
            'ride.c4',
            'ride.c5',
            'ride.c6',
            'ride.c7',
            'ride.c8',
            'ride.c9',
            'ride.c10',
            'ride.c11'
        ]

        join0 = joinRide.merge(self.riderTable, left_on='ride.riderId', right_on='rider.riderId')
        groupRider = join0.groupby(
            ['rider.riderId'],
        )[featureName].mean()

        groupRider['cluster'] = kmeans.predict(groupRider)

        endTime = time.time()

        groupRider.to_csv(
            'query8_gpu.csv',
            index=False,
        )
        return endTime - startTime - numbaTime

    def _query9(self):
        self._loadTables('query9')

        startTime = time.time()
        filterRide = self.rideReqTable[self.rideReqTable['rideReq.riderId'] == 0].head(1)

        (knnJoin, numbaTime) = self._spatialJoinKnn(
            filterRide,
            self.driverStatusTable,
            'rideReq.start',
            'drvStat.pos',
            100
        )

        knnJoin.sort_values('dist')

        endTime = time.time()

        knnJoin.to_csv(
            'query9_gpu.csv',
            index=False,
        )

        return endTime - startTime - numbaTime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genData', action='store_true')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--queries', nargs='+', help='select which queries you want')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if (args.genData):
        gen = QueryGen(args)
        gen.genData()
    
    if (args.gpu):
        gpu = QueryRunGpu(args)
        gpu._query1()
        gpu._query2()
        gpu._query3()
        gpu._query4()
        gpu._query5()
        gpu._query6()
        gpu._query7()
        gpu._query8()
        gpu._query9()

if __name__ == '__main__':
    main()
