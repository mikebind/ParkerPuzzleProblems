# For exploring Parker Alternative Puzzles
# import typing
from typing import Tuple, Optional, List, Dict, NamedTuple, Set, Iterable
import numpy as np
from enum import Enum

# from collections import namedtuple # it was better to inherit from typing.NamedTuple
# from bidict import bidict


# MARK: EdgePairSet
class EdgePairSet:
    """Exploring having a data structure which keeps track
    of edge pairings, so that it is easy to look up who
    is connected to whom.  Also, adding or removing
    an edge pairing should automatically add the
    complementary pair.  For example, if we connect
    +1 and -5, then we need to also pair +5 and 1-
    """

    _lookup: Dict[int, int]

    def __init__(self, lookupDict: Optional[Dict[int, int]] = None):
        if lookupDict is None:
            lookupDict = dict()
        self._lookup = lookupDict

    def addConnection(self, edge1: int, edge2: int):
        # Validate input
        if edge1 == edge2:
            raise EdgePairError("Not allowed to pair and edge with itself")
        priorKeys = tuple(self._lookup.keys())
        if (
            (edge1 in priorKeys)
            or (edge2 in priorKeys)
            or (-edge1 in priorKeys)
            or (-edge2 in priorKeys)
        ):
            raise EdgePairError(
                "Not allowed to overwrite the value associated with an existing key; only new pairs can be added with addConnection!"
            )
        self._lookup[edge1] = edge2
        self._lookup[edge2] = edge1
        # Also the complementary ones
        self._lookup[-edge1] = -edge2
        self._lookup[-edge2] = -edge1

    def removeConnection(self, edge1: int, edge2: int):
        for e in [edge1, edge2]:
            del self._lookup[e]
            del self._lookup[-e]

    def __getitem__(self, keyEdge):
        """Return None if given edge has no pair"""
        try:
            return self._lookup[keyEdge]
        except KeyError:
            return None

    def getComplimentaryPair(self, keyEdge):
        """Given an edge involved in a pair, return
        the other edges which must be linked.  For
        example if +1 and -5 are linked, then
        the complementary pair which must be linked
        are +5 and -1.
        """
        edge1 = keyEdge
        edge2 = self[keyEdge]
        edge1c = -edge1
        edge2c = -edge2
        return edge2c, edge1c

    def getFlatEdgeList(self) -> Tuple[int]:
        """Get a flattened list (tuple) of the edge
        numbers involved in all pairs
        """
        return tuple(self._lookup.keys())

    def __repr__(self):
        """Display summary in a concise way"""
        outStr = f"EdgePairSet({repr(self._lookup)})"
        return outStr

    def __str__(self):
        """Pretty print version of EdgePairSet"""
        absKeySet = set([np.abs(key) for key in self._lookup.keys()])
        pairs = [(key, self[key]) for key in absKeySet]
        linesOut = [
            "EdgePairSet Pairs: ",
            *[f"{pair[0]} to {pair[1]}" for pair in pairs],
        ]
        strOut = "\n  ".join(linesOut)
        return strOut

    def getDeepCopy(self) -> "EdgePairSet":
        """Return a deep copy of the current EdgePairSet instance"""
        copy = EdgePairSet()
        copy._lookup = {key: value for key, value in self._lookup.items()}
        return copy


# MARK: FragmentEdgePairSet
class FragmentEdgePairSet:
    _lookup: Dict[int, int]

    def __init__(self, lookupDict: Optional[Dict[int, int]] = None):
        if lookupDict is None:
            lookupDict = dict()
        self._lookup = lookupDict

    def __repr__(self):
        outStr = f"FragmentEdgePairSet(lookupDict={repr(self._lookup)})"
        return outStr

    def addConnection(self, edge1: int, edge2: int):
        if edge1 == edge2:
            raise EdgePairError("Not allowed to pair and edge with itself")
        priorKeys = tuple(self._lookup.keys())
        if (edge1 in priorKeys) or (edge2 in priorKeys):
            raise EdgePairError(
                "Not allowed to overwrite the value associated with an existing key; only new pairs can be added with addConnection!"
            )
        self._lookup[edge1] = edge2
        self._lookup[edge2] = edge1
        # But don't do the complementary ones -edge1 and -edge2

    def removeConnection(self, edge1: int, edge2: int):
        for e in [edge1, edge2]:
            del self._lookup[e]

    def __getitem__(self, keyEdge):
        """Return None if given edge has no pair"""
        try:
            return self._lookup[keyEdge]
        except KeyError:
            return None

    def extend(self, otherFragEdgePairs: "FragmentEdgePairSet"):
        """Add all edge pairs in the input other set of
        fragment edge pairs to self.
        """
        priorKeys = tuple(self._lookup.keys())
        for key, val in otherFragEdgePairs._lookup.items():
            if key in priorKeys:
                raise EdgePairError(
                    "Attempted to overwrite an existing pairing while extending! Not allowed!"
                )
            self._lookup[key] = val

    def getFlatEdgeList(self) -> Tuple[int]:
        """Get a flattened list (tuple) of the edge
        numbers involved in all pairs
        """
        return tuple(self._lookup.keys())

    def getDeepCopy(self) -> "EdgePairSet":
        """Return a deep copy of the current EdgePairSet instance"""
        copy = FragmentEdgePairSet()
        copy._lookup = {key: value for key, value in self._lookup.items()}
        return copy


# MARK: Enums
class PieceType(Enum):
    CORNER = 2
    BORDER = 1
    INTERIOR = 0


class EdgeClass(Enum):
    """Enum for puzzle piece edge types"""

    OUTER_BORDER = 0
    CW_FROM_CORNER = 1
    CCW_FROM_CORNER = 2
    CW_FROM_BORDER = 3
    CCW_FROM_BORDER = 4
    INTERIOR = 5


class GridCorner(Enum):
    """Enum for puzzle grid corners. Values are zero-based
    index, clockwise from northwest. (This allows easy
    calculations of rotations)
    """

    NW = 0
    NE = 1
    SE = 2
    SW = 3

    def __sub__(self, fromCorner: "GridCorner") -> int:
        """Allow calculation of rotation using "-" operator.
        gc1 - gc2 should return the rotation going from gc2
        to gc1
        """
        return GridCorner.getRotation(fromCorner, self)

    @classmethod
    def getRotation(cls, fromCorner: "GridCorner", toCorner: "GridCorner") -> int:
        """Get the rotation going from corner c1 to corner c2,
        in number of 90 deg steps.
        """
        rotation = (toCorner.value - fromCorner.value) % 4
        return rotation


partnerEdgeClasses: Dict[EdgeClass, Tuple[EdgeClass]] = {
    EdgeClass.CW_FROM_CORNER: (EdgeClass.CCW_FROM_BORDER,),
    EdgeClass.CCW_FROM_CORNER: (EdgeClass.CW_FROM_BORDER,),
    EdgeClass.CCW_FROM_BORDER: (EdgeClass.CW_FROM_CORNER, EdgeClass.CW_FROM_BORDER),
    EdgeClass.CW_FROM_BORDER: (EdgeClass.CCW_FROM_CORNER, EdgeClass.CCW_FROM_BORDER),
    EdgeClass.OUTER_BORDER: (),  # should this be None?
    EdgeClass.INTERIOR: (EdgeClass.INTERIOR,),
}


# Define a named tuple for anchor location info
class AnchorLocation(NamedTuple):
    pieceListIdx: int
    anchorGridPosition: Tuple[int, int]
    anchorOrientation: int

    def deepCopy(self):
        return AnchorLocation(
            self.pieceListIdx, self.anchorGridPosition, self.anchorOrientation
        )


class FragmentCoordinate(NamedTuple):
    rowCoord: int
    colCoord: int
    rotationCount: int

    # don't need deepcopy because already immutable


class FragmentCoordDict(dict):
    def deepCopy(self):  # -> "FragmentCoordDict"["PuzzlePiece", FragmentCoordinate]:
        copy = FragmentCoordDict({key: val for key, val in self.items()})
        return copy


# MARK: PuzzlePiece
class PuzzlePiece:
    def __init__(
        self,
        origRow: int,
        origCol: int,
        signedEdgeNums: Tuple[int, int, int, int],
    ):
        self.origPosition = (origRow, origCol)
        self._signedEdgeNums = signedEdgeNums
        self.pieceType = self.getPieceType()

    def __str__(self) -> str:
        lines = [
            "PuzzlePiece:",
            f"Original Location: {self.origPosition}",
            f"Edge Types: {self.getEdgeNums()}",
            f"Piece Type: {self.pieceType.name}",
        ]
        return "\n  ".join(lines)

    def __repr__(self) -> str:
        outStr = f"PuzzlePiece(origRow={self.origPosition[0]}, origCol={self.origPosition[1]}, signedEdgeNums={self._signedEdgeNums})"
        return outStr

    def getEdgeNums(self):
        return self._signedEdgeNums  # tuple(edge.edgeNum for edge in self.edges)

    def getPieceType(self) -> PieceType:
        """Determine piece type by kind and location of straight edges."""
        edgeNums = self.getEdgeNums()
        if len(edgeNums) != 4:
            raise WrongEdgeCountError(
                f"Pieces are expected to have exactly 4 edges, but this one has {len(edgeNums)} edges!"
            )
        numStraightEdges = np.sum(np.array(edgeNums) == 0)
        if numStraightEdges == 2:
            # The straight edges must be adjacent
            if (edgeNums[0] == 0 and edgeNums[2] == 0) or (
                edgeNums[1] == 0 and edgeNums[3] == 0
            ):
                raise InvalidPieceError(
                    "Invalid piece with straight edges on non-adjacent sides."
                )
            pieceType = PieceType.CORNER
        elif numStraightEdges == 1:
            pieceType = PieceType.BORDER
        elif numStraightEdges == 0:
            pieceType = PieceType.INTERIOR
        else:
            raise UnknownPieceTypeError(
                f"Unknown piece type with {numStraightEdges} edges."
            )
        return pieceType

    def getEdgeClasses(self) -> Tuple[EdgeClass]:
        """Return a tuple of the edge class associated with each edge"""
        edgeNums = self.getEdgeNums()
        pieceType = self.pieceType
        edgeClasses = []
        for dirIdx, edgeNum in enumerate(edgeNums):
            if edgeNum == 0:
                edgeClass = EdgeClass.OUTER_BORDER
            else:
                # Edge number is nonzero
                if pieceType == PieceType.INTERIOR:
                    edgeClass = EdgeClass.INTERIOR
                else:
                    # This piece has an outer border, the edge class
                    # depends on where this edge stands relative to
                    # outer boarder edge(s) and on the piece type
                    cwDirIdx = (dirIdx + 1) % 4
                    ccwDirIdx = (dirIdx - 1) % 4
                    cwFromBorder = edgeNums[cwDirIdx] == 0
                    ccwFromBorder = edgeNums[ccwDirIdx] == 0
                    if cwFromBorder:
                        edgeClass = (
                            EdgeClass.CW_FROM_BORDER
                            if pieceType == PieceType.BORDER
                            else EdgeClass.CW_FROM_CORNER
                        )
                    elif ccwFromBorder:
                        edgeClass = (
                            EdgeClass.CCW_FROM_BORDER
                            if pieceType == PieceType.BORDER
                            else EdgeClass.CCW_FROM_CORNER
                        )
                    else:
                        assert (
                            pieceType == PieceType.BORDER
                        ), "To get here, the piece type should only ever be a border edge piece!"
                        edgeClass = EdgeClass.INTERIOR
            # Store edge class
            edgeClasses.append(edgeClass)
        return tuple(edgeClasses)

    def getCWEdges(self, startIdx) -> Tuple[Tuple[int], Tuple[int]]:
        """Get the clockwise sequence of edges, starting from startIdx"""
        startIdx = startIdx % 4  # only 0-3 are valid
        edgeNums = self.getEdgeNums()
        cwEdges = (*edgeNums[startIdx:], *edgeNums[0:startIdx])
        return cwEdges


# MARK: FUNCTIONS
def rotateCoord(r, c, rotationOffset):
    """Rotate row,col coordinates by 90deg clockwise rotationOffset times"""
    rotatedRow, rotatedCol = r, c
    rotationOffset = rotationOffset % 4  # >3 loops
    for idx in range(rotationOffset):
        rotatedRow, rotatedCol = (rotatedCol, -rotatedRow)
    return rotatedRow, rotatedCol


def calcFragEdgePairs(
    edgePairs: EdgePairSet, pieceList: List[PuzzlePiece]
) -> FragmentEdgePairSet:
    """Filter the edge pairs (from a puzzle state) to include only those
    edges present in the pieces in the piece list.
    """
    statePairedEdges = edgePairs.getFlatEdgeList()
    # Loop over all piece edges and keep those which are in the supplied edgePairs
    edgesToInclude = [
        e for piece in pieceList for e in piece.getEdgeNums() if e in statePairedEdges
    ]
    fragEdgePairs = FragmentEdgePairSet()
    for e in edgesToInclude:
        partner = edgePairs[e]
        fragEdgePairs.addConnection(e, partner)
    # Note that this adds each edge twice, but it feels like it might be more
    # inefficent to try to filter it down to one pass, so I'm just going to
    # leave it.
    return fragEdgePairs


def calcFragmentCoord(
    anchorPiece: PuzzlePiece,
    anchorEdgeNum: int,
    anchorPieceCoord: FragmentCoordinate,
    partnerPiece: PuzzlePiece,
    partnerEdgeNum: int,
) -> FragmentCoordinate:
    """Calculate the relative fragment coordinate for the partner piece,
    based on the location of the anchor piece, its orientation, and the
    linked edges orientations.
    """
    # Determine which way the anchor edge is facing in the fragment
    # coordinate system
    originalOrientation = anchorPiece.getEdgeNums().index(anchorEdgeNum)
    anchorEdgeOrientation = (originalOrientation + anchorPieceCoord.rotationCount) % 4
    # The partner edge faces the opposite direction
    partnerEdgeOrientation = (anchorEdgeOrientation + 2) % 4
    # How rotated is this orientation from the original orientation?
    partnerEdgeOriginalOrientation = partnerPiece.getEdgeNums().index(partnerEdgeNum)
    partnerPieceRotationCount = (
        partnerEdgeOrientation - partnerEdgeOriginalOrientation
    ) % 4
    # Offsets
    N, E, S, W = ((-1, 0), (0, 1), (1, 0), (0, -1))
    offsets = (N, E, S, W)
    offset = offsets[anchorEdgeOrientation]
    partnerPieceRowCoord = anchorPieceCoord.rowCoord + offset[0]
    partnerPieceColCoord = anchorPieceCoord.colCoord + offset[1]
    newFragCoord = FragmentCoordinate(
        rowCoord=partnerPieceRowCoord,
        colCoord=partnerPieceColCoord,
        rotationCount=partnerPieceRotationCount,
    )
    return newFragCoord


def getOriginalCorner(piece: "PuzzlePiece") -> "GridCorner":
    """Given a puzzle piece, return the corresponding original
    grid corner enum for the piece.  If the input piece isn't a
    corner piece, a ValueError is raised
    """
    if piece.pieceType != PieceType.CORNER:
        raise ValueError("getOriginalCorner() called with non-corner piece!")
    origRow = piece.origPosition[0]
    origCol = piece.origPosition[1]
    if origRow == 0 and origCol == 0:
        gc = GridCorner.NW
    elif origRow == 0 and origCol != 0:
        gc = GridCorner.NE
    elif origRow != 0 and origCol == 0:
        gc = GridCorner.SW
    elif origRow != 0 and origCol != 0:
        gc = GridCorner.SE
    return gc


def findRequiredEdgePairs(puzzMapObj: "PuzzleMap"):
    """Return the list of required edge pairs given a puzzle map object
    (anchored or floating). This is purely geometric. If two pieces are
    adjacent in the map, then their shared side must be paired.
    """
    requiredEdgePairs = []
    # We should be able to do this in a vectorized way by working with
    # a boolean numpy array to identify N,S,E,W neighbors by shifting
    # array and overlapping.
    bmap = puzzMapObj.getBoolMap()
    # North-South overlaps
    hasSouthNeigbor = np.argwhere(np.logical_and(bmap[:-1, :], bmap[1:, :]))
    for row, col in hasSouthNeigbor:
        # Find south side of the piece at r,c
        northPiece = puzzMapObj.pieceMap[row, col]
        rot = puzzMapObj.rotationMap[row, col]
        # Get the current south side edge of the north piece
        edge1 = northPiece.getCWEdges(rot)[2]
        # Get the current north side edge of the south piece
        southPiece = puzzMapObj.pieceMap[row + 1, col]
        rot = puzzMapObj.rotationMap[row + 1, col]
        edge2 = southPiece.getCWEdges(rot)[0]
        requiredEdgePairs.append((edge1, edge2))
    # East-West overlaps
    hasEastNeighbor = np.argwhere(np.logical_and(bmap[:, :-1], bmap[:, 1:]))
    for row, col in hasEastNeighbor:
        # Find south side of the piece at r,c
        westPiece = puzzMapObj.pieceMap[row, col]
        rot = puzzMapObj.rotationMap[row, col]
        # Get the current south side edge of the north piece
        edge1 = westPiece.getCWEdges(rot)[1]
        # Get the current north side edge of the south piece
        eastPiece = puzzMapObj.pieceMap[row, col + 1]
        rot = puzzMapObj.rotationMap[row, col + 1]
        edge2 = eastPiece.getCWEdges(rot)[3]
        requiredEdgePairs.append((edge1, edge2))
    return requiredEdgePairs


# MARK: PuzzleFragment
class PuzzleFragment:
    """A representation of more than one puzzle piece, either with a fixed location (anchored),
    or floating. A puzzle fragment is invalid iff:
    * It extends outside the puzzle grid (either by being too long or, if anchored, extending
      outside the grid from where it is anchored)
    * It contains an invalid combination of pieces (i.e. an edge or interior piece where
      there must be a corner piece)
    A puzzle fragment can be defined by the set of pieces included, and by the set of connections
    linking them.
    """

    puzzleParameters: "PuzzleParameters"
    pieceList: List[PuzzlePiece]
    fragEdgePairs: FragmentEdgePairSet
    anchorLocation: Optional[AnchorLocation]
    puzzleMap: "PuzzleMap"
    _fragmentCoordDict: FragmentCoordDict
    _cachedFreeEdgesList: Optional[Tuple[int]]

    def __init__(
        self,
        puzzleParameters: "PuzzleParameters",
        pieceList: List[PuzzlePiece],
        fragEdgePairs: Optional[FragmentEdgePairSet],
        anchorLocation: Optional[AnchorLocation] = None,
        _fragmentCoordDict: Optional[
            FragmentCoordDict[PuzzlePiece, FragmentCoordinate]
        ] = None,
        puzzleMap: Optional["PuzzleMap"] = None,
        _cachedFreeEdgesList: Optional[Tuple[int]] = None,
    ):
        self.puzzleParameters = puzzleParameters
        self.anchorLocation = anchorLocation
        self.pieceList = pieceList
        self.fragEdgePairs = fragEdgePairs if fragEdgePairs else FragmentEdgePairSet()
        # ^^ Should we verify this is valid?? ^^
        if _fragmentCoordDict is None:
            self._fragmentCoordDict = FragmentCoordDict()
            self.assignFragmentCoordinates()
        else:
            # TODO: Validate this is OK first?
            self._fragmentCoordDict = _fragmentCoordDict
        # Bulding a puzzleMap depends on the fragmentCoordDict, so
        # this needs to happen after assignFragmentCoordinates()
        if puzzleMap is None:
            self.assignPuzzleMap()
        else:
            # Use input (should it be validated?)
            self.puzzleMap = puzzleMap
        # _cachedFreeEdgesList update depends on fragEdgePairs and
        # pieceList being filled in already
        self._cachedFreeEdgesList = _cachedFreeEdgesList
        if self._cachedFreeEdgesList is None:
            self.updateCachedFreeEdgesList()

    def __repr__(self) -> str:
        """Concise representation"""
        outStr = f"""PuzzleFragment(puzzleParameters={repr(self.puzzleParameters)}, pieceList={repr(self.pieceList)}, fragEdgePairs={repr(self.fragEdgePairs)}, anchorLocation={repr(self.anchorLocation)}, _fragmentCoordDict={repr(self._fragmentCoordDict)}, puzzleMap={repr(self.puzzleMap)})"""
        return outStr

    def getPuzzleSize(self):
        nRows = self.puzzleParameters.nRows
        nCols = self.puzzleParameters.nCols
        return nRows, nCols

    def assignPuzzleMap(self):
        if self.isAnchored():
            self.puzzleMap = AnchoredPuzzleMap(self.getPuzzleSize(), [self])
        else:
            self.puzzleMap = FloatingPuzzleMap(self.getPuzzleSize(), self)

    def getFragCoord(self, piece: PuzzlePiece) -> FragmentCoordinate:
        return self._fragmentCoordDict[piece]

    def setFragCoord(self, piece: PuzzlePiece, fragCoord: FragmentCoordinate):
        self._fragmentCoordDict[piece] = fragCoord

    def getMaxDim(self) -> int:
        """Get maximum dimension of the fragment"""
        minRow = 0
        maxRow = 0
        minCol = 0
        maxCol = 0
        for piece in self.pieceList:
            fc = self.getFragCoord(piece)
            minRow = np.min((minRow, fc.rowCoord))
            maxRow = np.max((maxRow, fc.rowCoord))
            minCol = np.min((minCol, fc.colCoord))
            maxCol = np.max((maxCol, fc.colCoord))
        rowDim = maxRow - minRow
        colDim = maxCol - minCol
        return max(rowDim, colDim)

    def getPieceFromEdge(self, edgeNum: int):
        return self.puzzleParameters.pieceFromEdgeDict[edgeNum]

    def addPiece(self, piece: PuzzlePiece):
        """Adding a layer here to detect if we try to add the same piece twice"""
        if piece in self.pieceList:
            raise ValueError("Piece already in this fragment's pieceList!!")
        self.pieceList.append(piece)

    def addPieces(self, pieces: Iterable[PuzzlePiece]):
        """Add a list of pieces"""
        for piece in pieces:
            self.addPiece(piece)

    def assignFragmentCoordinates(self):
        """Each piece within the fragment needs to be assigned local
        fragment coordinates.  Start from pieceList[0] and work out
        from there. When complete, all fragment pieces should have
        coordinates assigned.
        Initially, the starting piece is assigned a fragment coordinate
        of (0,0), but after the initial local coordinates are assigned,
        they will undergo one of two adjustments.  If the fragment
        has an anchor point, the local coordinates are converted to
        absolute coordinates by moving and rotating to put the anchor
        piece at the correct location and orientation in abs coords.
        If the fragment is floating, the local coordinates are adjusted
        to ensure that there are no negative coordinates, and furthermore,
        that there are no interior-only pieces on either the first row or
        first column.  This adjustment ensures that we can use local
        fragment coordinates directly for puzzle maps indices.
        """
        startingPiece = self.pieceList[0]
        # Determine it's initial starting fragment coordinate. If it already
        # has one, use that, if not, start at 0,0,0
        try:
            startingFragCoord = self.getFragCoord(startingPiece)
        except KeyError:
            # no existing fragment coord for this piece (brand new fragment)
            startingFragCoord = FragmentCoordinate(0, 0, 0)
            self.setFragCoord(startingPiece, startingFragCoord)
        piecesToCheckNext = set([startingPiece])
        # N, E, S, W = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # offsets = (N, E, S, W)
        minAssignedRow = 0
        minAssignedCol = 0
        checkedPieceList = []
        while len(piecesToCheckNext) > 0:
            piecesToCheckNow = piecesToCheckNext
            piecesToCheckNext = set()
            for piece in piecesToCheckNow:
                # check each edge and assign coords
                for edgeNum in piece.getEdgeNums():
                    # if edgeNum has a partner, assign the corresponding
                    # fragment coordinate to that piece associated with
                    # partner
                    partnerEdgeNum = self.fragEdgePairs[edgeNum]
                    if partnerEdgeNum:
                        # partner edge exists, get piece
                        partnerPiece = self.getPieceFromEdge(partnerEdgeNum)
                        # Calculate fragment coordinate
                        partnerCoord = calcFragmentCoord(
                            piece,
                            edgeNum,
                            self.getFragCoord(piece),
                            partnerPiece,
                            partnerEdgeNum,
                        )
                        # Check whether the partner piece already has a fragment coordinate
                        # If so, check that the current proposed coordinate doesn't conflict
                        # If it does throw an error, if it doesn't we're good, and don't
                        # add it to the list of pieces to check next (it's already been checked)
                        # If the partner piece does not have a fragment coordinate already, then
                        # assign it and add it to the list of pieces to check connections of next
                        try:
                            existingFragCoord = self.getFragCoord(
                                partnerPiece
                            )  # may throw KeyError
                            # Didn't throw error, continue to check if it matches
                            if existingFragCoord != partnerCoord:
                                raise InconsistentFragmentCoordinatesError(
                                    "Inconsistent! This piece already has fragment coordinates, but the new ones calculated don't agree!"
                                )
                        except KeyError:
                            # No current fragment coordinate, assign it
                            self.setFragCoord(partnerPiece, partnerCoord)
                            # Track if there is a new minimum assigned column or row
                            minAssignedRow = min(partnerCoord.rowCoord, minAssignedRow)
                            minAssignedCol = min(partnerCoord.colCoord, minAssignedCol)
                        if partnerPiece not in checkedPieceList:
                            # Add piece to the set of pieces to check the edges of next
                            piecesToCheckNext.add(partnerPiece)
                # Once a piece had been checked for partners once, we don't want to visit it again
                checkedPieceList.append(piece)
                # Finished looping over current list of pieces to check

        # Finished looping, all pieces should have consistent fragment coordinates assigned
        if not self.checkAllFragmentCoordinatesAssigned():
            raise LoosePiecesInFragmentError
        # Otherwise, everything is consistent and all pieces have been assigned local fragment coordinates
        # IF the fragment is anchored, I think we should convert all of these local
        # coordinates to true puzzle coordinates
        if self.isAnchored():
            # Convert local coord to absolute coord (we need to
            # force it on the initial assignment, because otherwise
            # it will appear that the intermediately applied local
            # coords are supposed to be absolute, and an error will
            # be raised)
            self.applyAnchorLocation(self.anchorLocation, forceFlag=True)
        else:
            # Not anchored, but may contain negative coords or non-border pieces in the
            # first row or column.  Translate minimally until this is no longer the case
            # Figure out what edge classes are in the minimum row and column facing up or
            # left. If we find non-outer boundary edge classes in either place, the offset
            # needs to be bumped one higher.
            # Example showing why we need to consider which direction an border edge is facing.
            # If we have floating fragment with a single corner piece and the open edges face
            # west and north, then its coordinate should be (1,1) rather than (0,0), but if
            # it faces east and south, then (0,0) is correct.
            bumpRows = False
            bumpCols = False
            for piece in self.pieceList:
                fc = self.getFragCoord(piece)
                edgeClasses = piece.getEdgeClasses()
                localNorthIdx = (0 + fc.rotationCount) % 4
                northEdgeClass = edgeClasses[localNorthIdx]
                if (
                    not bumpRows
                    and fc.rowCoord == minAssignedRow
                    and northEdgeClass != EdgeClass.OUTER_BORDER
                ):
                    bumpRows = True
                localWestIdx = (3 + fc.rotationCount) % 4
                westEdgeClass = edgeClasses[localWestIdx]
                if (
                    not bumpCols
                    and fc.colCoord == minAssignedCol
                    and westEdgeClass != EdgeClass.OUTER_BORDER
                ):
                    bumpCols = True
                if bumpCols and bumpRows:
                    break
            rowOffset = -minAssignedRow + (1 if bumpRows else 0)
            colOffset = -minAssignedCol + (1 if bumpCols else 0)
            # Apply the translation (definitely don't want repeated pieces here!)
            for piece in set(self.pieceList):
                fc = self.getFragCoord(piece)
                # update with new values
                newRow = fc.rowCoord + rowOffset
                newCol = fc.colCoord + colOffset
                newRot = fc.rotationCount
                newFc = FragmentCoordinate(newRow, newCol, newRot)
                self.setFragCoord(piece, newFc)

    def checkAllFragmentCoordinatesAssigned(self):
        """Return True if all pieces have fragment coordinates
        assigned, return False otherwise.
        """
        try:
            for piece in self.pieceList:
                coord = self.getFragCoord(piece)
        except KeyError:
            # A piece was not a key!
            return False
        # We made it, all pieces were keys
        return True

    def applyAnchorLocation(self, anchorLocation: AnchorLocation, forceFlag=False):
        """Apply a new anchor location to this fragment. Typically,
        this fragment should be floating if this is called, but
        if the new anchor location is the same as the old, then
        let it slide (with a possible logging warning?). If forceFlag is
        True, then the new anchor is applied without checking if it is
        consistent with the old one (if any).
        """
        anchorPiece = self.pieceList[anchorLocation.pieceListIdx]
        anchorRow, anchorCol = anchorLocation.anchorGridPosition
        anchorRotation = anchorLocation.anchorOrientation
        localFragCoord = self.getFragCoord(anchorPiece)
        # Local fragment coordinates should have been assigned already
        if localFragCoord is None:
            raise InconsistentFragmentCoordinatesError(
                "Missing fragment coordinate on anchor piece!"
            )
        # The new absolute fragment coordinate for the anchor piece is the anchor
        # location. Convert to a FragmentCoordinate to compare to the local frag coord
        newAbsFragCoord = FragmentCoordinate(anchorRow, anchorCol, anchorRotation)
        # If the fragment is already anchored and we are not forcing it, raise an error
        # unless the new anchored location for the anchor piecs already matches the prior
        # anchored location.
        if self.isAnchored() and (newAbsFragCoord != localFragCoord) and not forceFlag:
            raise InconsistentFragmentCoordinatesError(
                "Already anchored fragment is being assigned a new, inconsistent anchor!"
            )
        localRowCoord = localFragCoord.rowCoord
        localColCoord = localFragCoord.colCoord
        localRotation = localFragCoord.rotationCount
        rotationOffset = (anchorRotation - localRotation) % 4
        # For each piece, we need to apply the translation needed to move the anchorPiece
        # from its local coord to the origin, then apply the rotation, and then translate
        # the anchor piece to its new absolute coord
        for piece in self.pieceList:
            # Get old fragment coordinate, translate first then apply rotation
            oldFragCoord = self.getFragCoord(piece)
            # Apply anchor piece to origin translation
            translatedRowCoord = oldFragCoord.rowCoord - localRowCoord
            translatedColCoord = oldFragCoord.colCoord - localColCoord
            # Apply rotation
            # if the rotation offset is 1, the whole fragment should rotate clockwise
            # 90 deg around the origin.  So, a piece at (r,c) should end up at (c, -r)
            rotatedRowCoord, rotatedColCoord = rotateCoord(
                translatedRowCoord, translatedColCoord, rotationOffset
            )
            # Apply anchor piece to new absolute coord translation
            finalRowCoord = rotatedRowCoord + anchorRow
            finalColCoord = rotatedColCoord + anchorCol
            # Find new rotation count (old + offset)
            finalRotationCount = oldFragCoord.rotationCount + rotationOffset
            newFragCoord = FragmentCoordinate(
                finalRowCoord,
                finalColCoord,
                finalRotationCount,
            )
            # Replace with updated coordinate
            self.setFragCoord(piece, newFragCoord)
        # Add the new anchor location as the anchor location for this fragment
        self.anchorLocation = anchorLocation

    def applyRotation(self, rotationCount):
        """For a floating fragment, update all fragment coordinates
        and the puzzleMap by applying rotationCount 90deg rotations.
        This is most easily applied using the puzzleMap for this fragment,
        because that automatically means the tne new coordinates fit
        all the restrictions placed fragment coordinates
        """
        if self.isAnchored():
            raise ValueError(
                "This function is not supposed to be applied to anchored fragments, only floating!"
            )
        self.puzzleMap.applyRotation(rotationCount)
        # Update fragmentCoordDict from puzzleMap
        for piece in self.pieceList:
            fc = self.puzzleMap.getCoordFromPiece(piece)
            self.setFragCoord(piece, fc)
        # pieceList, fragEdgePairs, and free edges have not been changed

    def getFreeEdgesList(self):
        if self._cachedFreeEdgesList is None:
            self.updateCachedFreeEdgesList()
        return self._cachedFreeEdgesList

    def updateCachedFreeEdgesList(self):
        """Free edges are those which are on pieces in this fragment,
        but which are not in the list of edgePairs
        """
        flatPairedEdgeList = self.fragEdgePairs.getFlatEdgeList()
        flatFragmentEdgeSet = set()
        for piece in self.pieceList:
            for edgeNum in piece.getEdgeNums():
                flatFragmentEdgeSet.add(edgeNum)
        # The free edges are those which are in the fragment edges
        # but not in the paired edges
        freeEdges = (
            edgeNum
            for edgeNum in flatFragmentEdgeSet
            if edgeNum not in flatPairedEdgeList
        )
        self._cachedFreeEdgesList = tuple(freeEdges)

    def findFirstFreeEdge(
        self, startingPiece: Optional[PuzzlePiece] = None, searchDirection: int = 0
    ) -> Tuple[int, PuzzlePiece, int]:
        """Starting from north on the first piece in the pieceList,
        find the first edge which does not have a connected piece.
        If all edges are connected, then move to the next piece to the
        north and repeat.  Repeat until an edge is found, and then
        return a tuple of the free edge number, the puzzle piece it
        was found on, and the direction it was on that piece (in the
        local fragment coordinate system).
        """
        # Default to first piece on list
        if startingPiece is None:
            startingPiece = self.pieceList[0]
        # We will use the piece from edge dict and list of free edges repeatedly
        freeEdges = self.getFreeEdgesList()
        # Initialize before loop starts
        curPiece = startingPiece
        numPiecesChecked = 0
        while numPiecesChecked <= len(self.pieceList):
            # Get the local rotation of the current piece (in fragment coord system)
            localRot = self.getFragCoord(curPiece).rotationCount
            # Then the edge numbers clockwise starting from the search direction
            edgeNums = curPiece.getCWEdges((localRot + searchDirection) % 4)
            # Check each edge, stop at the first free one
            for rotFromSearch, edgeNum in enumerate(edgeNums):
                if edgeNum in freeEdges:
                    firstFreeEdgeNum = edgeNum
                    firstFreeEdgePiece = curPiece
                    mostRecentEdgeDirection = searchDirection + rotFromSearch
                    # Return whenever the first free edge is found
                    return firstFreeEdgeNum, firstFreeEdgePiece, mostRecentEdgeDirection
            # No free edges found on this piece, find connected piece in the search direction
            edgeNumInSearchDirection = edgeNums[0]
            pieceInSearchDirection = self.parentPuzzleState.getPieceFromEdge(
                self.fragEdgePairs[edgeNumInSearchDirection]
            )
            # Keep the search direction, update the current piece
            curPiece = pieceInSearchDirection
            numPiecesChecked += 1
        # This process should always eventually terminate for sane inputs, and never
        # reach this line, but to avoid a bug or bad input causing an infinite loop,
        # we leave the loop if we have checked more pieces than there are in this fragment
        raise FirstFreeEdgeError(
            "No free edges found!! This should be impossible, check edgePairs for problems?"
        )

    def checkEdgeFree(self, edgeNum):
        """Edge is free if it is not in the fragment edge pair set"""
        return self.fragEdgePairs[edgeNum] is None

    def getEdgeDirection(self, edgeNum: int) -> int:
        """Get the direction an edge faces in the fragment coordinate grid"""
        if edgeNum == 0:
            raise ZeroEdgeNumberNotUniqueError(
                "Tried to check the edge direction of a non-unique edge number (0)!"
            )
        piece = self.parentPuzzleState.getPieceFromEdge(edgeNum)
        coord = self.getFragCoord(piece)
        edgeDirection = piece.getCWEdges(coord.rotationCount).index(edgeNum)
        return edgeDirection

    def getOrderedExternalEdgeList(
        self, verifyFlag: bool = True
    ) -> List[List[Tuple[int, PuzzlePiece]]]:
        """Identify a starting free (unconnected) edge.  Then, follow the open
        edges clockwise around the entire fragment. If verifyFlag, then
        do the extra check to make sure all edges are either in the list of
        ordered external edges, or in the edgePairs.  (True for now,
        may change to false later to speed computation after some testing)

        Search strategy: Once the first free edge is found, store it in the externalEdgeList.
        Then, check the next edge clockwise around the current piece.
        If a piece is there, move to that piece and check the same edge direction
        as the most recent external edge.  Either there is a piece there or it is
        open.  If open, add it to the list and check the next clockwise edge
        """
        boolMap = self.puzzleMap.getBoolMap()
        pieceMap = self.puzzleMap.pieceMap
        rotationMap = self.puzzleMap.rotationMap

        # Rewrite firstFreeEdgeNum to use pieceMap?

        # The search pattern is different for finding the first edge
        (
            firstFreeEdgeNum,
            firstFreeEdgePiece,
            mostRecentEdgeDirection,
        ) = self.findFirstFreeEdge(startingPiece=self.pieceList[0])
        # Start the list of external edges
        externalEdgeList = [(firstFreeEdgeNum, firstFreeEdgePiece)]
        # Need edge number AND piece because 0 edge numbers are not unique mappings
        # and we want to be able to tell whether we have completed the circuit
        # Now search for the next edge
        currentPiece = firstFreeEdgePiece
        # The search direction is one step clockwise around the current piece
        searchDirection = (mostRecentEdgeDirection + 1) % 4
        while not self.externalEdgeListClosedFlag(externalEdgeList):
            # nextFreeEdge, mostRecentEdgeDirection, nextPiece = self.findNextFreeEdge(currentPiece, mostRecentEdgeDirection)
            edgeNumToCheck = currentPiece.getEdgeNums()[searchDirection]
            if self.checkEdgeFree(edgeNumToCheck):
                if (edgeNumToCheck, currentPiece) in externalEdgeList[1:]:
                    raise ValueError(
                        "Algorithm error, repeating an edge/piece combination"
                    )
                    # actually, this can happen on a corner piece, with two zero edges (both
                    # of which will results in (0, piece) entries.  We could add the edgeclass
                    # as a distinguisher, but I wonder if it is better to just use the puzzleMap
                    # now that we have developed that. )
                # Edge is free, add it to the list
                externalEdgeList.append((edgeNumToCheck, currentPiece))
                # Keep current piece, update most recent edge direction
                mostRecentEdgeDirection = searchDirection
                searchDirection = (searchDirection + 1) % 4
            else:
                # Edge not free, move to the connected piece in the search
                # direction
                partnerEdge = self.fragEdgePairs[edgeNumToCheck]
                partnerPiece = self.puzzleParameters.getPieceFromEdge(partnerEdge)
                currentPiece = partnerPiece
                # Update the search direction ccw one step
                searchDirection = (searchDirection - 1) % 4
        # External edge list is now a complete closed loop
        # TODO: what about the case where there is a closed loop but a hole in the middle?
        # Such as when the outer border is finished!  Also, we could start in a hole, and
        # never get to the outside this way.  Need to handle these topological quirky situations
        # Could probably just start at the first remaining edge and then make another closed loop,
        # and repeat until there are no more non-included edges.

        # Verify if requested. Are all edges accounted for?
        if verifyFlag:
            for piece in self.pieceList:
                for edge in piece.getEdgeNums():
                    inFreeEdges = (edge, piece) in externalEdgeList
                    inEdgePairs = self.fragEdgePairs[edge] is not None
                    if not inFreeEdges and not inEdgePairs:
                        raise GetOrderedEdgesVerificationError(
                            f"Edge {edge} is in neither the external nor the paired edge lists!"
                        )

        return [externalEdgeList[:-1]]  # drop the repeated edge before returning

    def findNextFreeEdge(self, currentPiece: PuzzlePiece, mostRecentEdgeDirection: int):
        """Find next free edge, NOT FUNCTIONAL, ignore for now"""
        searchDirection = (mostRecentEdgeDirection + 1) % 4
        edgeNumToCheck = currentPiece.getEdgeNums()[searchDirection]
        if self.checkEdgeFree(edgeNumToCheck):
            # Edge is free, add it to the list
            nextFreeEdge = (edgeNumToCheck, currentPiece)
            # Keep current piece, update most recent edge direction
            mostRecentEdgeDirection = searchDirection
            return nextFreeEdge, mostRecentEdgeDirection
        else:
            # Edge not free, move to the connected piece in the search
            # direction
            partnerEdge = self.edgePairs[edgeNumToCheck]
            partnerPiece = self.parentPuzzleParamObj.pieceFromEdgeDict[partnerEdge]
            currentPiece = partnerPiece
            # Update the search direction ccw one step
            searchDirection = (searchDirection - 1) % 4

    def externalEdgeListClosedFlag(
        self, orderedEdgeList: List[Tuple[int, PuzzlePiece]]
    ) -> bool:
        """Check that the list makes a closed loop"""
        tooLong = len(orderedEdgeList) > 4 * len(self.pieceList)
        if tooLong:
            raise RuntimeError(
                "An impossible number of edges has been added to the external edge list!"
            )
        longEnough = len(orderedEdgeList) > 1
        endsMatch = orderedEdgeList[0] == orderedEdgeList[-1]
        return longEnough and endsMatch

    def isValid(self) -> bool:
        """Check fragment validity, return True or False. Valid if it fits in the current
        parentPuzzleGrid.  This can depend on whether it is anchored or not (for example,
        a horziontal strip might be known to be invalid even if it would fit vertically)
        """
        # TODO TODO TODO Decide on how I want to do the validity checking before really
        # implementing this
        if self.isAnchored():
            # just check if any puzzle piece in the fragment is located outside the grid
            # Actually, there are more ways to be invalid... wrong piece type connected
            # for example?
            pass

    def isAnchored(self) -> bool:
        """Check whether the fragment is anchored or not"""
        return not (self.anchorLocation is None)

    def deepCopy(self) -> "PuzzleFragment":
        pieceListCopy = [*self.pieceList]
        anchorLocCopy = self.anchorLocation.deepCopy() if self.anchorLocation else None
        coordDictCopy = self._fragmentCoordDict.deepCopy()
        puzzMapCopy = self.puzzleMap.deepCopy()
        copy = PuzzleFragment(
            self.puzzleParameters,  # ok to use orig
            pieceList=pieceListCopy,
            fragEdgePairs=self.fragEdgePairs.getDeepCopy(),
            anchorLocation=anchorLocCopy,
            _fragmentCoordDict=coordDictCopy,
            puzzleMap=puzzMapCopy,
        )
        return copy


# What about the idea of a puzzleState?
""" A complete puzzle state could be derived from
* the puzzle dimensions (MxN)
    * We can derive the original pieces and original connections from this
* the set of piece connections and anchored fragment or piece locations assigned so far for
  the alternate solution
* a list of edges yet to try (or a list of already tried edges)

Next, we need a function to try adding a proposed connection and check if it leads to a contradiction.
(Including whether it's complementary connection leads to a contradiction). If both are valid, we 
create a new derived puzzleState and keep going until we hit a contradiction, or until we find a 
solution. 
We also need a strategy for choosing the next edge to connect.  We'll start with anchoring one corner, and
work our way around the edge until the edge is closed. We'll always try the next numerically and type-eligible edge.
"""

""" I think there are two main ways we could approach the next stage. We have a PuzzleState,
and we want to try adding a new connection. We could 
1) Add the connection and then check for any internal invalidity in the resulting PuzzleState
OR
2) Go through all the consequences of adding the connection ahead of time, and only actually
generate the resulting PuzzleState if it will be valid.  
Option 1 feels safer, but I feel like there could be a lot of unnecessary repeated calculations
which it would be nice to avoid.  For example, if the new connection adds to the width of the
puzzle, then there is no way that it could cause the height to become too large, but if we 
are just checking a state for validity, we would be checking both the height and the width. 

Perhaps it would be helpful to enumerate all the ways a connection could be invalid here:
** A piece connects to itself
* A fragment connects to itself in two different places (it's OK if it works out geometrically)
**? A piece is in two different fragments at the same time
* A piece is both in the loose pieces and in a fragment at the same time
* An edge is paired with more than one other edge
* An edge is paired with its negative (original connection)
* A piece repeats its original location and orientation (except the 0,0 location)
* A piece of the wrong type is placed in an anchored fragment where it can't fit, e.g.
  * a corner piece placed in a border or interior spot, 
  * a border piece in a corner or interior spot
  * an interior piece in a corner or border spot
  * a border piece oriented with the flat side not towards the puzzle border
  * a corner piece oriented incorrectly
* A floating fragment is too large to fit in the puzzle (e.g. has a linear dimension
  which is longer than either puzzle dimension)
* There is no conceivable way that all the puzzle fragments could simultaneously fit 
  within the puzzle grid because there is no way for the shapes to fit together. I don't
  know whether this will be worth checking or not.  It sounds potentially hard to code, and
  if it doesn't work, it may become more obvious soon as more connections are added. 

Some of the above conditions seem like they will be best applied to narrow down the edges
we would try. This is great because then we may have fewer conditions to check afterwards.

Others, it seems like it might make more sense to check after applying the potential connection.
** for check before trying
*** for check after

I would like to get clear about the steps:
I have a puzzle state, I have next edge to connect to
I then need to make a list of possible edges to try.
That list should be narrowed down to be as short as possible
* narrowed by piece type (corner, border, interior)
* narrowed by absolute location (can't be an edge on the piece which was originally there)
Anything else ahead of time?
Loop over those edges and try them, reject if:
* connecting fragments which end up outside the puzzle
* if complementary connection is
    * self connection
    * makes invalid floating fragment

Code to write next: 
* find the next edge to connect to (first one on ordered free edges of anchored fragment
  which is not numbered 0)
* define edge types (cw_from_border, ccw_from_border, cw_from_corner, ccw_from_corner, interior, outerBorder) make easy lookup
* define desired edge type from anchored location of next edge to connect to
* narrow edge list from all free edges to the proper type of edge
    * consider narrowing based on complimentary edge? (harder?) 
* try it! then complimentary connection.  Check for invalidity... If valid, return new puzzle state


"""

# MARK: PuzzleMap
""" I would like to create a class which makes an informative grid of data
about puzzle piece arrangment.  These should be arrays which hold puzzle
pieces and orientations in a spatial grid.  Like fragments, these should
come in two flavors, anchored and floating. Anchored maps are on the true 
puzzle grid, use absolute coordinates, and can have multiple contributing
fragments.  Floating maps are single-fragment and use a variation of local
coordinates. The rotation of Anchored maps is fixed 
"""


class PuzzleMap:
    # Parent class for shared methods and properties of Anchored  and
    # floating puzzle maps
    pieceMap: np.ndarray
    rotationMap: np.ndarray
    puzzleSize: Tuple[int, int]
    _coordFromPiece: Dict[PuzzlePiece, FragmentCoordinate]

    def __init__(self, puzzleSize):
        self.puzzleSize = puzzleSize

    def buildMaps(self):
        """Override in children"""
        pass

    def getBoolMap(self) -> np.ndarray:
        """Get a boolean map of where pieces are"""
        boolMap = np.logical_not(np.equal(self.pieceMap, None))
        return boolMap

    def isAnchored(self) -> bool:
        """Override in children"""
        pass

    def applyRotation(self, rotationCount) -> None:
        """Override in children"""
        pass

    def getCoordFromPiece(self, piece: PuzzlePiece) -> FragmentCoordinate:
        return self._coordFromPiece[piece]

    def deepCopy(self) -> "PuzzleMap":
        """Reimplement in child classes"""
        pass


class AnchoredPuzzleMap(PuzzleMap):
    def __init__(
        self,
        puzzleSize,
        fragments: Optional[List[PuzzleFragment]] = None,
        pieceMap: Optional[np.ndarray] = None,
        rotationMap: Optional[np.ndarray] = None,
        _coordFromPiece: Optional[Dict] = None,
    ):
        super().__init__(puzzleSize)
        if (pieceMap is None) or (rotationMap is None) or (_coordFromPiece is None):
            # If any of these are None, build from fragments
            if fragments is None:
                raise ValueError(
                    "Fragments was None when maps needed to be built from them in AnchoredPuzzleMap!"
                )
            pieceMap, rotationMap, _coordFromPiece = self.buildMaps(fragments)
        self.pieceMap = pieceMap
        self.rotationMap = rotationMap
        self._coordFromPiece = _coordFromPiece

    def __repr__(self):
        outStr = f"AnchoredPuzzleMap(puzzleSize={self.puzzleSize}, pieceMap={repr(self.pieceMap)}, rotationMap={repr(self.rotationMap)}, _coordFromPiece={repr(self._coordFromPiece)})"
        return outStr

    def buildMaps(self, anchoredFragments: List[PuzzleFragment]):
        """Build the piece map, rotation map, and _coordFromPiece dict
        TODO: Should this include consistency checks? I am leaving them
        out for now but could possibly use this to replace sections of
        checkGeometry(), where the consistency checks are the point...
        """
        pieceMap = np.full(self.puzzleSize, None, dtype=object)
        rotationMap = np.full_like(pieceMap, 0, dtype=int)
        coordFromPieceDict = dict()
        # like puzzleGridArray in checkGeometry!  add code from there
        for frag in anchoredFragments:
            for piece in frag.pieceList:
                fragCoord = frag.getFragCoord(piece)
                row = fragCoord.rowCoord
                col = fragCoord.colCoord
                rot = fragCoord.rotationCount
                pieceMap[row, col] = piece
                rotationMap[row, col] = rot
                # Store lookup in dict
                coordFromPieceDict[piece] = fragCoord

        return pieceMap, rotationMap, coordFromPieceDict

    def isAnchored(self):
        return True

    def applyRotation(self, rotationCount):
        raise TypeError("AnchoredPuzzleMaps should not be rotated!")

    def deepCopy(self) -> "AnchoredPuzzleMap":
        """Make a deep copy"""
        copy = AnchoredPuzzleMap(
            self.puzzleSize,
            pieceMap=self.pieceMap.copy(),
            rotationMap=self.rotationMap.copy(),
            _coordFromPiece=self._coordFromPiece.copy(),
        )
        return copy


class FloatingPuzzleMap(PuzzleMap):
    def __init__(
        self,
        puzzleSize: Tuple[int, int],
        fragment: Optional[PuzzleFragment] = None,
        pieceMap: Optional[np.ndarray] = None,
        rotationMap: Optional[np.ndarray] = None,
        _coordFromPiece: Optional[Dict[PuzzlePiece, FragmentCoordinate]] = None,
    ):
        super().__init__(puzzleSize)
        if (pieceMap is None) or (rotationMap is None) or (_coordFromPiece is None):
            # If any of these are None, build from fragment
            if fragment is None:
                raise ValueError(
                    "Fragment was None when maps needed to be built from it in FloatingPuzzleMap!"
                )
            pieceMap, rotationMap, _coordFromPiece = self.buildMaps(fragment)
        self.pieceMap = pieceMap
        self.rotationMap = rotationMap
        self._coordFromPiece = _coordFromPiece

    def __repr__(self):
        outStr = f"FloatingPuzzleMap(puzzleSize={self.puzzleSize}, pieceMap={repr(self.pieceMap)}, rotationMap={repr(self.rotationMap)}, _coordFromPiece={repr(self._coordFromPiece)})"
        return outStr

    def buildMaps(
        self, fragment: PuzzleFragment
    ) -> Tuple[np.ndarray, np.ndarray, Dict[PuzzlePiece, FragmentCoordinate]]:
        """Build the piece map, rotation map, and coordFromPiece dict"""
        coordFromPiece = dict()
        # First, figure out what size the maps need to be for this floating
        # fragment. Fragment coordinates already ensure that only border and
        # corner pieces can be on row 0 and column 0.
        """We can start by finding the mximum assigned row coord and
        maximum assigned column coord.  
        """

        maxAssignedRow = 0
        maxAssignedCol = 0
        for piece in fragment.pieceList:
            fc = fragment.getFragCoord(piece)
            maxAssignedRow = max(maxAssignedRow, fc.rowCoord)
            maxAssignedCol = max(maxAssignedCol, fc.colCoord)
        # If the last row or column has an interior only piece, then we need
        # to bump the map out one more row or col because we know it must be there
        bumpRows = False
        bumpCols = False
        for piece in fragment.pieceList:
            fc = fragment.getFragCoord(piece)
            edgeClasses = piece.getEdgeClasses()
            localSouthIdx = (2 + fc.rotationCount) % 4
            southEdgeClass = edgeClasses[localSouthIdx]
            if (
                (not bumpRows)
                and (fc.rowCoord == maxAssignedRow)
                and (southEdgeClass != EdgeClass.OUTER_BORDER)
            ):
                bumpRows = True
            localEastIdx = (1 + fc.rotationCount) % 4
            eastEdgeClass = edgeClasses[localEastIdx]
            if (
                not bumpCols
                and fc.colCoord == maxAssignedCol
                and (eastEdgeClass != EdgeClass.OUTER_BORDER)
            ):
                bumpCols = True
            if bumpCols and bumpRows:
                break
        maxMapRowIdx = maxAssignedRow + (1 if bumpRows else 0)
        maxMapColIdx = maxAssignedCol + (1 if bumpCols else 0)
        # Allocate maps
        mapArraySize = (maxMapRowIdx + 1, maxMapColIdx + 1)
        pieceMap = np.full(mapArraySize, None, dtype=object)
        rotationMap = np.full_like(pieceMap, 0, dtype=int)
        for piece in fragment.pieceList:
            fc = fragment.getFragCoord(piece)
            pieceMap[fc.rowCoord, fc.colCoord] = piece
            rotationMap[fc.rowCoord, fc.colCoord] = fc.rotationCount
            # Store coordinate lookup in dict
            coordFromPiece[piece] = fc
        return pieceMap, rotationMap, coordFromPiece

    def isAnchored(self):
        return False

    def applyRotation(self, rotationCount):
        """Apply rotationCount 90 deg rotations to each of the maps
        in this puzzleMap. And update the rotationMap counts appropriately.
        Finally, update the coordinate from piece dict.
        """
        ccwRotCount = -rotationCount  # numpy rotates ccw, while we want cw
        self.pieceMap = self.pieceMap.rot90(ccwRotCount)
        self.rotationMap = self.rotationMap.rot90(ccwRotCount)
        self.rotationMap = (self.rotationMap + rotationCount) % 4
        # Then update the other puzzlemap component
        _coordFromPiece = dict()
        # Loop over all rows and columns
        mapShape = self.pieceMap.shape
        for r in range(mapShape[0]):
            for c in range(mapShape[1]):
                piece = self.pieceMap[r, c]  # a piece or None
                if piece is not None:
                    fc = FragmentCoordinate(r, c, self.rotationMap[r, c])
                    _coordFromPiece[piece] = fc
        self._coordFromPiece = _coordFromPiece
        return

    def deepCopy(self):
        """Make a deep copy"""
        copy = FloatingPuzzleMap(
            self.puzzleSize,
            pieceMap=self.pieceMap.copy(),
            rotationMap=self.rotationMap.copy(),
            _coordFromPiece=self._coordFromPiece.copy(),
        )
        return copy


# MARK: PuzzleState
class PuzzleState:
    def __init__(
        self,
        parent: "PuzzleParameters",
        edgePairs: EdgePairSet,
        fragments: List[PuzzleFragment],
        loosePieces: List[PuzzlePiece],
    ):
        self.parent = parent
        self.edgePairs = edgePairs
        self.fragments = fragments
        self.loosePieces = loosePieces
        self._edgePairsSyncNeeded: bool = False

    def __repr__(self):
        outStr = f"PuzzleState(parent={repr(self.parent)}, edgePairs={repr(self.edgePairs)}, fragments={repr(self.fragments)}, loosePieces={repr(self.loosePieces)})"
        return outStr

    def deepCopy(self) -> "PuzzleState":
        """Return a copy of the current puzzle state.
        The parent does not need to be copied. The fragments
        and loose pieces themselves are OK to not make
        fresh copies (pieces don't change and unmodified
        fragments don't need to be duplicated), but the
        lists containing them should be fresh ones because
        if we, for example, remove a piece from the list of
        loose pieces, we want that change to happen ONLY
        in the copied state, not in the state it was copied
        from.
        I've rethought this, and decided that fragments should
        be copied at puzzle state creation time. They could
        possibly be modified multiple times while a single
        puzzle state is being explored, and it doesn't make
        sense to keep making new fragments at every modification.
        """
        copy = PuzzleState(
            self.parent,
            edgePairs=self.edgePairs.getDeepCopy(),
            fragments=[f.deepCopy() for f in self.fragments],
            loosePieces=[p for p in self.loosePieces],
        )
        # Sync state can't be set in constructor, but should
        # probably be passed on.
        copy._edgePairsSyncNeeded = self._edgePairsSyncNeeded
        return copy

    def addConnection(
        self, edgesToPair: Tuple[int, int], recursionCount: int = 0
    ) -> "PuzzleState":
        """Starting from the current state, try to add the edge pair connection
        given. If invalid for any reason, throw an AddConnectionError exception.
        If valid, return a new PuzzleState which incorporates the new connection.
        Each new connection involves linking either a fragment to another fragment,
        a fragment to a loose piece, or two loose pieces together.
        Through all manipulations here, we need to make sure not to alter any of the
        present puzzle state's variables, because if there is a problem in this or
        any deeper connection, we need to be able to restore the last state where
        we hadn't found a contradiction.

        Before entering this function, possible new edge pairs should have been
        narrowed down to exclude those which:
         * are already part of a pair
         * aren't of incompatible edge classes
         * are only on possible piece types (for anchored fragments only so far)

        This function must detect all remaining contradiction types:
        * A fragment self-connects in a non-geometrically allowable way
        * A piece self-connects
        * A non-anchored fragment nevertheless cannot fit into the puzzle
        * The other edges to pair cause any problems (same list, but less
          prescreening so also need to check for problems there)

        In addition this function must enforce any additional edge pair requirements
        which are entailed.  For example, if placing a piece means that it must
        geometrically connect with another piece which has already been placed in
        the current puzzle state, then we must enforce adding that edgePair connection
        as well (probably by recursively calling this function with a new state).
        """
        edge1 = edgesToPair[0]
        edge2 = edgesToPair[1]

        newPuzzleState = self.deepCopy()

        newPuzzleState.joinEdgesAndCompEdges(edge1, edge2)

        # Do geometrical checks on new puzzle state
        newPuzzleState.findAndAssignNewAnchors()
        requiredNewEdgePairs = newPuzzleState.identifyRequiredNewEdgePairs()

        while requiredNewEdgePairs:
            for edgePair in requiredNewEdgePairs:
                newPuzzleState.joinEdgesAndCompEdges(*edgePair)
            newPuzzleState.findAndAssignNewAnchors()
            newPuzzleState.checkGeometry()  # TODO: consider, here or after while?
            requiredNewEdgePairs = newPuzzleState.identifyRequiredNewEdgePairs()

        # Check if

        # Any additional new edge pairs required by the geometrical arrangement
        # of pieces? If so, we need to test adding them before returning

        return newPuzzleState

    def joinEdgesAndCompEdges(self, edge1: int, edge2: int):
        """Join given edges, their complementary edge pair, and update
        the edgePairSet. This method bundles the joinEdges() calls so that
        the edgePairs is kept in sync and does not remain unsynced for long.
        """
        self.joinEdges(edge1, edge2)
        self.joinEdges(-edge1, -edge2)
        self.edgePairs.addConnection(edge1, edge2)
        self._edgePairsSyncNeeded = False

    def identifyRequiredNewEdgePairs(self) -> Tuple[Tuple[int, int]]:
        """Placed pieces may geometrically entail other connections. This function
        is supposed to determine those. The basic approach will be to make
        puzzle grid maps and make a list of all required edges, and compare with
        the list of already specified edges, and return any that aren't already
        specified. All anchored fragments can share one map. Each floating fragment
        will need its own map and its own coordinate system.
        """
        anchoredFragments = [f for f in self.fragments if f.isAnchored()]
        floatingFragments = [f for f in self.fragments if not f.isAnchored()]
        # Anchored fragments have a common puzzle map
        anchPuzzleMap = AnchoredPuzzleMap(self.getPuzzleSize(), anchoredFragments)
        reqEdgePairs = findRequiredEdgePairs(anchPuzzleMap)
        # Floating fragments are dealt with individually
        for fragment in floatingFragments:
            fragReqEdgePairs = findRequiredEdgePairs(fragment.puzzleMap)
            reqEdgePairs.extend(fragReqEdgePairs)
        # Now, we can make a new list containing only the required edge pairs which are
        # not already in the EdgePairSet we have
        newReqEdgePairs = []
        for e1, e2 in reqEdgePairs:
            if self.edgePairs[e1] is None and self.edgePairs[e2] is None:
                # Neither edge is in the current EdgePairSet, this is a new required pairing!
                newReqEdgePairs.append((e1, e2))
            elif self.edgePairs[e1] != e2:
                # At least one of e1 and e1 is in the set, but not paired with the other
                raise AddConnectionError(
                    "Geometrically required edge pairing does not match existing pairing in EdgePairSet!"
                )
        # Return the new edge pairs we need to add
        return newReqEdgePairs

    # TODO: also remember that perhaps new fragment copies need to be made
    # whenever a fragment is created or modified?? Or is that no longer the
    # case because we are always working with the new puzzle state and its
    # consequences?? If exceptions are always only caught at the solvePuzzle
    # level, and they always lead to trying another new edge pair, that triggers
    # a new test PuzzleState, so we don't need to worry about later modifications
    # if we make a copy at that point.

    def findAndAssignNewAnchors(self):
        """This function tries to anchor floating fragments by process of
        elimination if possible.  Currently, it only functions on
        corner pieces, if there are only one or two which remain unanchored.
        """
        # Corner assignment
        # If two or fewer corners remain unanchored, see if you can anchor
        # them by process of elimination.
        cornerPieces = self.parent.piecesFromPieceType[PieceType.CORNER]
        cornerFragments = [self.getFragmentFromPiece(p) for p in cornerPieces]
        # Determine if each corner is anchored
        anchoredMask = np.array(
            [(frag.isAnchored() if frag else False) for frag in cornerFragments]
        )

        nCornersAssigned = np.sum(anchoredMask)
        floatingIdxs = np.flatnonzero(np.logical_not(anchoredMask))
        anchoredIdxs = np.flatnonzero(anchoredMask)
        assignedGridCorners = []
        for idx in anchoredIdxs:
            anchCornerPiece = cornerPieces[idx]
            frag = cornerFragments[idx]
            fc = frag.getFragCoord(anchCornerPiece)
            assignedGridCorners.append(
                self.gridCornerFromCoord(fc.rowCoord, fc.colCoord)
            )
        unassignedGridCorners = [
            gc for gc in GridCorner if gc not in assignedGridCorners
        ]
        unassignedCornerPieces = [cornerPieces[idx] for idx in floatingIdxs]
        # Try to anchor any floating corners, if possible
        if nCornersAssigned == 4:
            # All corners already assigned, nothing to do
            pass
        elif nCornersAssigned == 3:
            # 3 already assigned, the 4th one is known by process of elimination
            unassignedCornerPiece = unassignedCornerPieces[0]
            unassignedGridCorner = unassignedGridCorners[0]
            # Assign the unassigned corner piece to the unassigned grid corner
            self.anchorCornerPiece(unassignedCornerPiece, unassignedGridCorner)
        elif nCornersAssigned == 2:
            # Two corners assigned, two unassigned.  If either unaassigned corner piece
            # has an original corner position which matches either unassigned corner,
            # then we can assign both to where they must go.
            origCornersForUnassCornerPieces = [
                getOriginalCorner(p) for p in unassignedCornerPieces
            ]
            # Check for any overlap between
            intersection = set(unassignedCornerPieces).intersection(
                origCornersForUnassCornerPieces
            )
            if intersection:
                # There is overlap, we can assign the corners
                overlapGC = intersection[0]
                # This corner is both the original corner for one of the unassigned
                # corner pieces AND is one of the unoccupied corners on the puzzle
                # grid. That means that the corner piece MUST be assigned to the
                # OTHER unoccupied corner.
                otherUnoccupiedCorner = [
                    ugc for ugc in unassignedGridCorners if ugc != overlapGC
                ][0]
                overlapPiece = [
                    ucp
                    for ucp, oc in zip(
                        unassignedCornerPieces, origCornersForUnassCornerPieces
                    )
                    if oc != overlapGC
                ][0]
                otherPiece = [p for p in unassignedCornerPieces if p != overlapPiece][0]
                # Anchor overlapPiece to otherUnoccupiedCorner
                self.anchorCornerPiece(overlapPiece, otherUnoccupiedCorner)
                # Anchor otherPiece to overlpGC
                self.anchorCornerPiece(otherPiece, overlapGC)
            # if we got this far without assigning, then we can't assign corners in this
            # case, OK to move on.

        # Corner assignment is finished
        # TODO: attempt other anchoring procedures here.  However, given how
        # complicated it was to just do to corners, maybe I'll stop here for now,
        # assuming that any inevitable contradictions will come up quickly enough
        # even if it is possible to anticipate them at this point.

    def anchorCornerPiece(self, cornerPiece: PuzzlePiece, gridCorner: GridCorner):
        """Apply an anchor the given corner piece to the given grid corner.
        The corner piece may be a loose piece or already on a fragment.
        If a loose piece, convert to an anchored fragment.  If on a floating
        fragment, anchor it to the corner.  If on an anchored fragment, check
        that the assigned coordinate is consistent (if not raise an error).
        """
        origGridCorner = getOriginalCorner(cornerPiece)
        rotCount = gridCorner - origGridCorner
        if rotCount == 0:
            raise AddConnectionError(
                "When assigning new anchors, a corner was forced to it's original location!"
            )
        # Need the coordinates for the assigned grid corner
        gridRow, gridCol = self.getGridCornerCoordinates(gridCorner)
        fragment = self.getFragmentFromPiece(cornerPiece)
        if fragment:
            # This corner piece is already on a fragment
            anchorPieceIdx = fragment.pieceList.index(cornerPiece)
        else:
            # This corner piece is a loose piece, convert to a new fragment
            # and then anchor
            anchorPieceIdx = 0
            fragment = PuzzleFragment(
                self.parent,
                pieceList=[cornerPiece],
                fragEdgePairs=FragmentEdgePairSet(),
            )
            # Need to add this new fragment to the puzzle state
            self.fragments.append(fragment)
            # Need to remove the corner piece from the loose pieces list
            self.loosePieces.remove(cornerPiece)
        # Construct new anchor location object
        newAnchorLocation = AnchorLocation(
            pieceListIdx=anchorPieceIdx,
            anchorGridPosition=(gridRow, gridCol),
            anchorOrientation=rotCount,
        )
        # Apply the new anchor
        fragment.applyAnchorLocation(newAnchorLocation)

    def getGridCornerCoordinates(self, gridCorner: GridCorner) -> Tuple[int, int]:
        """Get the grid row and column assocated with the give grid corner"""
        nRows, nCols = self.getPuzzleSize()
        cornerCoordDict = {
            GridCorner.NW: (0, 0),
            GridCorner.NE: (0, nCols - 1),
            GridCorner.SE: (nRows - 1, nCols - 1),
            GridCorner.SW: (nRows - 1, 0),
        }
        return cornerCoordDict[gridCorner]

    def getPuzzleSize(self):
        return self.parent.nRows, self.parent.nCols

    def gridCornerFromCoord(self, row, col) -> GridCorner:
        """Get grid corner enum from a row and column of grid position.
        Raises ValueError if the coordinates are not a corner
        """
        nRows, nCols = self.getPuzzleSize()
        if (row not in (0, nRows - 1)) or (col not in (0, nCols - 1)):
            raise ValueError(
                f"({row}, {col}) is not a corner coordinate and is therefore invalid input to gridCornerFromCoord()!"
            )
        if row == 0 and col == 0:
            gc = GridCorner.NW
        elif row == 0 and col != 0:
            gc = GridCorner.NE
        elif row != 0 and col == 0:
            gc = GridCorner.SW
        elif row != 0 and col != 0:
            gc = GridCorner.SE
        return gc

    def checkGeometry(self):
        """Check the puzzle state for invalidity based on geometrical
        considerations. (Piece positions, floating fragment sizes, process of
        elimination reasoning). Geometrically required edge connections are
        handled separately from this function.
        """
        nRows, nCols = self.getPuzzleSize()
        maxRowInd, maxColInd = nRows - 1, nCols - 1
        anchoredFragments = [f for f in self.fragments if f.isAnchored()]
        floatingFragments = [f for f in self.fragments if not f.isAnchored()]
        # An Anchored fragment can have each of its pieces checked
        # to see whether it is in its original location and orientation.
        # If so (unless it is at 0,0) then an error should be thrown
        anchPuzzMap = AnchoredPuzzleMap(self.getPuzzleSize(), anchoredFragments)
        for frag in anchoredFragments:
            for piece in frag.pieceList:
                origRow, origCol = (*piece.origPosition,)
                newCoord = anchPuzzMap.getCoordFromPiece(piece)
                newRow, newCol = (newCoord.rowCoord, newCoord.colCoord)
                if (
                    (origCol == newCol)  # same col
                    and (origRow == newRow)  # same row
                    and newCoord.rotationCount == 0  # unrotated
                ) and not ((origRow, origCol) == (0, 0)):
                    raise AddConnectionError(
                        "Piece illegally placed in identical position and orientation!"
                    )
                # If any piece on an anchored fragment has a coordinate less than zero
                # or greater than the max, that is a contradiction
                if newRow < 0 or newCol < 0 or newRow > maxRowInd or newCol > maxColInd:
                    raise AddConnectionError(
                        "Anchored piece located outside puzzle grid!"
                    )

        # puzzleGridArray is a boolean 2d array in the shape of the puzzle, with True where
        # anchored pieces sit.
        maxDim = self.getPuzzleMapMaxOpenDim(anchPuzzMap.getBoolMap())
        # For floating fragments, we can check if any dimension is greater than the remaining
        # available space dimensions on the puzzle.

        for frag in floatingFragments:
            # Check basic dimensions
            fragMaxDim = frag.getMaxDim()
            if fragMaxDim > maxDim:
                raise AddConnectionError(
                    "Floating fragment has too large dimension to possibly fit in the remaining puzzle space!"
                )
        # TODO: Consider the following, but decide if worth implementing or NOT!
        # Additionally, we could check that they could actually fit somewhere on the puzzle by
        # scanning all possible positions and orientations (stopping when we hit one which works)
        # We could also pay attention to whether there are any edge or corner pieces involved in
        # the fragment, which would greatly restrict the possible places which are eligible to
        # work. If no edge or corner pieces, then we can still restrict possible places to those
        # with NO edge/corner pieces.
        # TODO: Consider, is it better to scan here, or is it better to just wait for contradictions
        # to arise.  Checking seems like it might be a bit computationally intensive, so maybe it
        # is better to just wait for the more obvious failure one or two steps down the line?
        #

    def getPuzzleMapMaxOpenDim(self, puzzleMap: np.ndarray) -> int:
        """Get the maximum open dimension left in the puzzle. For now, this will be
        very simple minded, but it (TODO) could be made more sophistcated over time.
        """
        # Currently just sum of open spaces in each column or row
        maxHeight = np.max(np.sum(puzzleMap, axis=0))
        maxWidth = np.max(np.sum(puzzleMap, axis=1))
        maxDim = np.max([maxHeight, maxWidth])
        return maxDim

    def joinEdges(
        self,
        edge1,
        edge2,
    ) -> None:
        """Called only from a new puzzle state copy, joinEdges updates
        the fragments and loose pieces according to the new edges to
        join. Several types of contradictions can be detected at this
        point, and they will lead to raising AddConnectionError
        exceptions. The puzzle state edgePairs will be out of sync
        after a call to joinEdges, and must be updated after all
        joinEdges calls are made.  A boolean flag of
        self._edgePairsSyncNeeded is set to True by calls to joinEdges
        to mark this state.
        Nothing is returned, rather the calling new puzzle state copy
        is updated.
        We do not update the puzzleState edgePairs within this function
        because the one call to joinEdges before a call for the
        implied complementary pair cannot be consistently represented
        in an EdgePairSet. Only after both pair joins have been completed
        can we update the EdgePairSet.
        """
        if edge1 == edge2:
            raise AddConnectionError("Illegal to join edge to itself!")
        # Gather some elements used in multiple cases below
        frag1 = self.getFragmentFromEdge(edge1)  # None if on loose piece
        frag2 = self.getFragmentFromEdge(edge2)  # None if on loose piece
        piece1 = self.getPieceFromEdge(edge1)
        piece2 = self.getPieceFromEdge(edge2)

        if piece1 == piece2:
            raise AddConnectionError("Illegal piece self join!")
        if (frag1 is None) and (frag2 is None):
            # Neither edge is on an existing fragment.  Join the two
            # loose pieces to create a new fragment
            fragEdgePairs = FragmentEdgePairSet()
            fragEdgePairs.addConnection(edge1, edge2)
            newFrag = PuzzleFragment(
                self.parent, [piece1, piece2], fragEdgePairs, anchorLocation=None
            )
            self.fragments.append(newFrag)
            self.loosePieces.remove(piece1)
            self.loosePieces.remove(piece2)
        elif frag1 == frag2:
            # new edges are on the same fragment, double-check if it is OK
            if not self.intraFragmentConnnectionOK(frag1, edge1, edge2):
                raise AddConnectionError("Illegal fragment self join in joinEdges!")
            # Otherwise OK to join.  Minimal changes.  Same fragments with the same
            # piece lists, anchor location, fragment coordinate dictionary; just add one
            # edge pair
            frag1.fragEdgePairs.addConnection(edge1, edge2)
            frag1.updateCachedFreeEdgesList()
            # No change in loose piece list
        elif (frag1 is not None) and (frag2 is not None):
            # The two edges are each on different fragments, we need to join the two
            # fragments into a single new fragment
            self.joinFragments(edge1, frag1, edge2, frag2)
        else:
            # One is on a fragment and the other is on a loose piece,
            # join the loose piece to the fragment
            if frag1 is None:
                # edge2 is on fragment, edge1 on loose piece
                frag = frag2
                piece = self.getPieceFromEdge(edge1)
                # otherFrags = [f for f in fragments if f != frag2]
            else:
                # edge1 is on fragment, edge2 on loose piece
                frag = frag1
                piece = self.getPieceFromEdge(edge2)
                # otherFrags = [fragm.deepCopy() for fragm in fragments if fragm != frag1]
            self.loosePieces.remove(piece)
            frag.addPiece(piece)
            frag.fragEdgePairs.addConnection(edge1, edge2)
            frag.assignFragmentCoordinates()
            frag.assignPuzzleMap()
            frag.updateCachedFreeEdgesList()

        # The puzzle state edgePairs is now out of sync
        self._edgePairsSyncNeeded = True
        return

    def joinFragments(
        self, edge1: int, frag1: PuzzleFragment, edge2: int, frag2: PuzzleFragment
    ) -> None:
        """Join edge1 on fragment1 to edge2 on fragment2."""
        # If both are anchored, ensure that edge1 and edge2 already abut one another,
        # if not, raise an error
        piece1 = self.getPieceFromEdge(edge1)
        piece2 = self.getPieceFromEdge(edge2)
        frag1Coord = frag1.getFragCoord(piece1)
        frag2Coord = frag2.getFragCoord(piece2)
        if frag1.isAnchored() and frag2.isAnchored():
            # The partner location from edge1 must be the location of piece2
            # and the partner location from edge2 must be the location of
            # piece1.
            reqFrag2Coord = calcFragmentCoord(piece1, edge1, frag1Coord, piece2, edge2)
            reqFrag1Coord = calcFragmentCoord(piece2, edge2, frag2Coord, piece1, edge1)
            if frag1Coord != reqFrag1Coord or frag2Coord != reqFrag2Coord:
                raise AddConnectionError(
                    "Fragments to join have incompatible anchored coordinates!"
                )
            # If we pass that test, then these fragments already have compatible coordinates,
            # and we can just add all the info from frag2 to frag1, and then remove frag2 from the
            # fragment list
            frag1.addPieces(frag2.pieceList)
            frag1.fragEdgePairs.extend(frag2.fragEdgePairs)
            frag1.fragEdgePairs.addConnection(edge1, edge2)  # don't forget the new pair
            for piece in frag2.pieceList:
                frag1.setFragCoord(piece, frag2.getFragCoord(piece))
            frag1.assignPuzzleMap()  # could also merge these, but straight forward just to rebuild
            frag1.updateCachedFreeEdgesList()
            # Remove fragment 2
            self.fragments.remove(frag2)
        elif frag1.isAnchored() or frag2.isAnchored():
            # One of them is anchored and the other is floating
            if frag1.isAnchored():
                aFrag, aEdge, aPiece, aCoord = frag1, edge1, piece1, frag1Coord
                fFrag, fEdge, fPiece, fCoord = frag2, edge2, piece2, frag2Coord
            else:
                aFrag, aEdge, aPiece, aCoord = frag2, edge2, piece2, frag2Coord
                fFrag, fEdge, fPiece, fCoord = frag1, edge1, piece1, frag1Coord

            # Rotate and determine translation of the floating one to align it with the anchored one
            reqFPieceCoord = calcFragmentCoord(aPiece, aEdge, aCoord, fPiece, fEdge)
            fPieceCoordBeforeRotation = fFrag.getFragCoord(fPiece)
            rotOffset = (
                reqFPieceCoord.rotationCount - fPieceCoordBeforeRotation.rotationCount
            ) % 4
            fFrag.applyRotation(rotOffset)
            fPieceCoordAfterRotation = fFrag.getFragCoord(fPiece)
            rOffset = reqFPieceCoord.rowCoord - fPieceCoordAfterRotation.rowCoord
            cOffset = reqFPieceCoord.colCoord - fPieceCoordAfterRotation.colCoord
            # Add pieces from the floating fragment to the anchored fragment
            aFrag.addPieces(fFrag.pieceList)
            # Transfer edge pairs also
            aFrag.fragEdgePairs.extend(fFrag.fragEdgePairs)
            # Add the new connection
            aFrag.fragEdgePairs.addConnection(edge1, edge2)
            for piece in fFrag.pieceList:
                fcOld = fFrag.getFragCoord(piece)
                # apply translation
                fcNew = FragmentCoordinate(
                    fcOld.rowCoord + rOffset,
                    fcOld.colCoord + cOffset,
                    fcOld.rotationCount,  # already rotated, so no need to apply offset here
                )
                aFrag.setFragCoord(piece, fcNew)
            # Update the puzzleMap
            aFrag.assignPuzzleMap()
            aFrag.updateCachedFreeEdgesList()
            # Remove the floating one
            self.fragments.remove(fFrag)
        else:
            # both floating fragments
            # Rotate frag2 coord until edge2 faces edge1, apply needed translation to place
            # edge2 in contact with edge1, then apply any needed global translations
            reqPiece2Coord = calcFragmentCoord(
                piece1, edge1, frag1.getFragCoord(piece1), piece2, edge2
            )
            rotOffset = (
                reqPiece2Coord.rotationCount - frag2.getFragCoord(piece2).rotationCount
            ) % 4
            frag2.applyRotation(rotOffset)
            curCoord = frag2.getFragCoord(piece2)
            rOffset = reqPiece2Coord.rowCoord - curCoord.rowCoord
            cOffset = reqPiece2Coord.colCoord - curCoord.colCoord
            # Apply changes to frag1
            frag1.addPieces(frag2.pieceList)
            frag1.fragEdgePairs.extend(frag2.fragEdgePairs)
            frag1.fragEdgePairs.addConnection(edge1, edge2)
            for piece in frag2.pieceList:
                fcOld = frag2.getFragCoord(piece)
                fcNew = FragmentCoordinate(
                    fcOld.rowCoord + rOffset,
                    fcOld.colCoord + cOffset,
                    fcOld.rotationCount,  # already rotated, no offset needed
                )
                frag1.setFragCoord(piece, fcNew)
            # Rebuild puzzleMap()
            frag1.assignPuzzleMap()
            frag1.updateCachedFreeEdgesList()
            # Remove fragment 2
            self.fragments.remove(frag2)
        return

    def intraFragmentConnnectionOK(
        self, frag1: PuzzleFragment, edge1: int, edge2: int
    ) -> bool:
        """Check whether a proposed intra-fragment connection works out
        geometrically.  The necessary and sufficient conditions are:
        * edge1 and edge2 have to point in opposite directions in the fragment
          coordinate system
        * stepping from the piece with edge1 in the direction of edge1, must land
          you on the coordinates of the piece with edge2.
        """
        edge1Direction = frag1.getEdgeDirection(edge1)
        edge2Direction = frag1.getEdgeDirection(edge2)
        # First check: If the edges don't face each other, the connection can't be OK
        if ((edge1Direction + 2) % 4) != edge2Direction:
            return False
        # Second Check: One step in the edge1 direction must lead you to the edge2 piece coord
        piece1 = self.getPieceFromEdge(edge1)
        piece2 = self.getPieceFromEdge(edge2)
        coord1 = frag1.fragmentCoordDict[piece1]
        coord2 = frag1.fragmentCoordDict[piece2]
        # N,E,S,W
        rowOffset = (-1, 0, 1, 0)[edge1Direction]
        colOffset = (0, 1, -1, 0)[edge1Direction]

        partnerPieceRow = coord1.rowCoord + rowOffset
        partnerPieceCol = coord1.colCoord + colOffset
        if not (
            (partnerPieceRow == coord2.rowCoord)
            and (partnerPieceCol == coord2.colCoord)
        ):
            return False
        # If we get here, they are adjacent and facing, this intrafragment connection is OK
        return True

    def selectActiveEdge(self) -> int:
        """Select the next active edge to connect to
        This should be a signed integer.
        """
        activeFragment = None
        # Find the first active fragment
        for fragment in self.fragments:
            if fragment.isAnchored():
                activeFragment = fragment
                break
        if activeFragment is None:
            raise Exception("No anchored fragment!")
        edgeLoops = activeFragment.getOrderedExternalEdgeList()
        # Find the first nonzero edge
        activeEdge = None
        for edgeList in edgeLoops:
            for edge, piece in edgeList:
                if edge != 0:
                    activeEdge = edge
                    return activeEdge
        # No nonzero edges found, raise an exception
        raise Exception("No nonzero edges found on the active fragment")

    def findCandidateEdges(self, activeEdge: int) -> Tuple[int]:
        """Given the current puzzle state, and the active
        edge which we are trying to connect to, return as small
        a list of candidate edges as possible.
        """
        # The first cut is determined by the type of edge of the active edge
        activeEdgeClass = self.getEdgeClass(activeEdge)
        candidateEdgeClasses = self.getComplementaryEdgeClasses(activeEdgeClass)
        candidateEdges = []
        for edgeClass in candidateEdgeClasses:
            candidateEdges.extend(self.getEdgesFromEdgeClass(edgeClass))
        # Remove any edges which are already paired
        alreadyPairedEdges = self.edgePairs.getFlatEdgeList()
        candidateEdges = [
            edge for edge in candidateEdges if edge not in alreadyPairedEdges
        ]
        # Remove the edge which is paired with the activeEdge in the original puzzle
        if -activeEdge in candidateEdges:
            candidateEdges.remove(-activeEdge)
        # Narrow down by piece type (geometrical considerations, especially for
        # anchored fragments)
        allowedPieceTypes = self.getAllowedPieceTypes(activeEdge)
        candidateEdges = [
            edge
            for edge in candidateEdges
            if self.getPieceTypeFromEdge(edge) in allowedPieceTypes
        ]
        # Similar checks should be run for the implied next edge pair
        candidateEdgesToRemove = []
        for edge in candidateEdges:
            impliedPair = [-edge, -activeEdge]
            edgeClasses = [self.getEdgeClass(e) for e in impliedPair]
            pieceTypes = [self.getPieceTypeFromEdge(e) for e in impliedPair]
            allowedPieceTypes = [self.getAllowedPieceTypes(e) for e in impliedPair]

            # Are edge classes compatible?
            if (
                edgeClasses[0] not in self.getComplementaryEdgeClasses(edgeClasses[1])
            ) or (
                edgeClasses[1] not in self.getComplementaryEdgeClasses(edgeClasses[0])
            ):
                # -activeEdge is not the right edge class to connect to -edge or vice versa
                candidateEdgesToRemove.append(edge)
            elif -edge in self.edgePairs.getFlatEdgeList():
                # I think this is redundant...
                candidateEdgesToRemove.append(edge)
            elif (pieceTypes[1] not in allowedPieceTypes[0]) or (
                pieceTypes[0] not in allowedPieceTypes[1]
            ):
                # Implied pieces are not compatible
                candidateEdgesToRemove.append(edge)
        # Remove any edges disallowed by the implied pair
        candidateEdges = [e for e in candidateEdges if e not in candidateEdgesToRemove]

        return tuple(candidateEdges)

    def getPieceTypeFromEdge(self, edgeNum: int) -> PieceType:
        """Get Piece type from edge number"""
        if edgeNum == 0:
            raise ZeroEdgeNumberNotUniqueError(
                "Can't get piece type from zero edge number because it is not unique!"
            )
        piece = self.getPieceFromEdge(edgeNum)
        return self.parent.pieceTypeFromPiece[piece]

    def getAllowedPieceTypes(self, edgeNum: int) -> Tuple[PieceType]:
        """Based on geometrical considerations, what piece type or types could
        go in the location complementary to the given edge.  This is
        straightforward to discern if the active edge is on an anchored
        fragment, but can sometimes still be done for floating fragments.
        For loose pieces, we will not narrow down any further than the
        edge class-based considerations which are carried out separately
        from this method (so we can just return all piece types here without
        further consideration).
        """
        piece = self.getPieceFromEdge(edgeNum)
        fragment = self.getFragmentFromEdge(edgeNum)
        # We can skip the rest of the method if the piece is loose
        if fragment is None:
            # Loose piece, no restriction on purely geometrical grounds
            # (edgeClass grounds considered separately)
            allowedPieceTypes = (PieceType.BORDER, PieceType.CORNER, PieceType.INTERIOR)
            return allowedPieceTypes
        # If there is a fragment, we will want to know the coordinates of the new
        # partner piece, and will want to be able to compare to puzzle sizes
        nRows, nCols = self.getPuzzleSize()

        coord = fragment.getFragCoord(piece)
        origEdgeDir = piece.getEdgeNums().index(edgeNum)
        activeEdgeDir = coord.rotationCount + origEdgeDir
        # N,E,S,W
        rowOffset = (-1, 0, 1, 0)[activeEdgeDir]
        colOffset = (0, 1, -1, 0)[activeEdgeDir]
        partnerPieceRow = coord.rowCoord + rowOffset
        partnerPieceCol = coord.colCoord + colOffset

        if fragment.isAnchored():
            # For anchored fragments, the coordinate tells us the piece type
            onHorizBorder = (partnerPieceRow == 0) or (partnerPieceRow == nRows - 1)
            onVertBorder = (partnerPieceCol == 0) or (partnerPieceCol == nCols - 1)
            if (not onHorizBorder) and (not onVertBorder):
                allowedPieceTypes = (PieceType.INTERIOR,)
            elif onHorizBorder and onVertBorder:
                allowedPieceTypes = (PieceType.CORNER,)
            else:
                # Not interior or corner, must be on border
                allowedPieceTypes = (PieceType.BORDER,)
        else:
            # Floating fragment
            """For floating fragments, the situation is more complicated.
            By the way local fragment coordinates are assigned, we can
            tell interior vs border pieces to some extent.  A piece on
            the rim of the current fragment puzzleMap is always a corner or border
            piece. A piece in the interior is always an interior piece.
            However, a new piece location on the rim might be either a
            border or interior piece (an interior piece would trigger
            expanding the puzzleMap grid after adding). In addition,
            we don't know if the floating fragment is oriented correctly,
            so if the puzzle is not square, we may not know whether the
            next piece along a given border is another edge piece or
            a corner piece (if we are reaching the corner distance in
            one dimension but not the other).
            """
            fragNRows, fragNCols = fragment.puzzleMap.pieceMap.shape
            onFragHorizBorder = (partnerPieceRow == 0) or (
                partnerPieceRow == fragNRows - 1
            )
            onFragVertBorder = (partnerPieceCol == 0) or (
                partnerPieceCol == fragNCols - 1
            )
            if (not onFragHorizBorder) and (not onFragVertBorder):
                # Partner piece location is on a floating fragment, and not
                # on the outer rim of the fragment puzzleMap grid.  Therefore
                # it must be an interior piece, because it is in a region
                # known to contain only interior pieces.
                allowedPieceTypes = (PieceType.INTERIOR,)
            elif onFragHorizBorder and onFragVertBorder:
                # On corner of floating grid.  Piece type cannot be interior,
                # because it must connect to an existing border piece along
                # a border, but it will require more work to determine if we can narrow
                # it down to be only corner or only border.
                """This depends on if we know where a true corner is. If we
                have a corner piece included, or if we have border pieces included
                on two adjacent sides, then we know where a true corner is.
                """
                allowedPieceTypes = (PieceType.BORDER, PieceType.INTERIOR)
                trueCornerKnown = False  # placeholder
                if trueCornerKnown:
                    """Find the partner piece offset from the true corner.
                    If it is (0,0), then the partner must be a corner piece.
                    If it is (maxPuzzleDim-1,minPuzzleDim-1) or
                    (minPuzzleDim-1,maxPuzzleDim-1), then it must be a corner
                    piece.
                    If it is along the same border as the known corner (i.e.
                    if rowOffset or colOffset = 0), then it is:
                    * definitely Border if the other offset is < minPuzzleDim-1
                    * either Border or Corner if == minPuzzleDim-1
                    * definitely Border if >=minPuzzleDim and < maxPuzzleDim-1
                    * definitely Corner if == maxPuzzleDim
                    If it is not on the same border (minOffSet and maxOffSet)
                    both > 0, it must still be on the same border as at least
                    one Border piece (because there is no way to get to the
                    corner of the existing grid otherwise!, no diagonal
                    connections!). In this case, we know a remote corner, and
                    a local border, which means we know the exact dimension of
                    one side. Therefore, we know the exact dimension of the
                    OTHER side (the one we're on), and where we fall on it, so
                    we know whether we are Border or Corner:

                    """
                    pass
                else:
                    """No corner known.  However, we must have an edge on
                    our border that we connect to (no diagonal connections!).
                    We must be in one of the following
                    conditions:
                    * We have an edge on our side t
                    """
                    pass
            elif onFragHorizBorder or onFragVertBorder:
                """The partner piece location is on the outer rim of the
                fragment puzzleMap grid, but not on a corner.  This is
                the most complex case, because it could be that this is
                the first piece location on that edge of the rim, and
                it may be the case that it should be an interior piece
                and the grid should grow, or it could be the case that
                it is a border piece and the grid should not grow. However
                we CAN rule out it being a corner piece, because when
                a corner piece is added, it must always be added at the
                a corner of the fragment grid, and this location is already
                NOT at a corner. So, at least for now, we will leave this
                case as allowing border or interior, but not corner piece types.
                """
                allowedPieceTypes = (PieceType.BORDER, PieceType.INTERIOR)
            else:
                # Should be unreachable
                raise RuntimeError("Reached case which should be unreachable!")

        return allowedPieceTypes

    def getFragmentFromEdge(
        self,
        edgeNum: int,
    ) -> Optional[PuzzleFragment]:
        """Find the fragment the given edge is on.  Return None if there is
        no fragment which has this edge (or maybe throw an error?)
        """
        activePiece = self.getPieceFromEdge(edgeNum)
        fragment = self.getFragmentFromPiece(activePiece)
        return fragment  # can be None

    def getFragmentFromPiece(self, piece: PuzzlePiece) -> Optional[PuzzleFragment]:
        """Find the fragment that the given piece is on. Return None
        if the piece is a loose piece.
        """
        if piece in self.loosePieces:
            return None
        else:
            for fragment in self.fragments:
                if piece in fragment.pieceList:
                    return fragment
        # Not in any fragment or in loose piece list!
        raise RuntimeError("Input piece is neither loose nor in any fragment")

    def getEdgeClass(self, edgeNum: int) -> EdgeClass:
        return self.parent.edgeClassFromEdge[edgeNum]

    def getPieceFromEdge(self, edgeNum: int) -> PuzzlePiece:
        return self.parent.pieceFromEdgeDict[edgeNum]

    def getEdgesFromEdgeClass(self, edgeClass: EdgeClass) -> List[int]:
        return self.parent.edgesFromEdgeClass[edgeClass]

    def getComplementaryEdgeClasses(self, edgeClass: EdgeClass) -> EdgeClass:
        return partnerEdgeClasses[edgeClass]

    def getPieceList(self) -> Tuple[PuzzlePiece]:
        return self.parent.pieceList

    def isComplete(self) -> bool:
        """Return true if the puzzle state represents
        a complete puzzle.  For now we identify that by there
        being no more loose pieces and only one fragment. Should
        there be any other conditions?"""
        zeroLoosePieces = len(self.loosePieces) == 0
        onlyOneFragment = len(self.fragments) == 1
        return zeroLoosePieces and onlyOneFragment


# MARK: PuzzParameters
class PuzzleParameters:
    def __init__(self, nRows: int, nCols: Optional[int] = None):
        if nCols is None:
            nCols = nRows
        self.nRows = nRows
        self.nCols = nCols
        self.pieceList = self.generatePieces()
        (
            self.pieceFromEdgeDict,
            self.pieceTypeFromPiece,
            self.piecesFromPieceType,
            self.edgesFromEdgeClass,
            self.edgeClassFromEdge,
        ) = self.classifyPiecesAndEdges()
        self.initPuzzleState = self.generateInitialPuzzleState()

    def __repr__(self):
        outStr = f"PuzzleParameters(nRows={self.nRows}, nCols={self.nCols})"
        return outStr

    def generatePieces(self) -> List[PuzzlePiece]:
        """Create Puzzle Pieces in initial orientation with initial unique edge numberings
        Edge numbering rules:
          Outer flat edges are of type 0 and then vertical edges are numbered in reading
          order, then horizontal edges are numbered in reading order.  Polarity is assigned
          such that the left or upper side of the edge has polarity +1, while the right or
          lower side has polarity -1.  Straight edges are of type 0 and have polarity 0.
        """
        nRows = self.nRows
        nCols = self.nCols

        pieceList = []
        method1 = False
        for rIdx in range(nRows):
            for cIdx in range(nCols):
                if method1:
                    # number exterior edges (then change to 0s)
                    W = rIdx * (nCols + 1) + cIdx
                    E = W + 1
                    firstN = nRows * (nCols + 1)
                    N = firstN + cIdx + rIdx * nCols
                    S = N + nCols
                else:
                    # number only interior edges
                    firstN = (nCols - 1) * (nRows - 1) + nCols
                    N = firstN + nCols * (rIdx - 1) + cIdx
                    S = N + nCols
                    W = (nCols - 1) * rIdx + cIdx
                    E = W + 1

                Np, Ep, Sp, Wp = (-1, 1, 1, -1)  # default polarities
                # Apply straight edges on the boundaries
                if rIdx == 0:
                    N = 0
                    Np = 0
                elif rIdx == nRows - 1:
                    S = 0
                    Sp = 0
                if cIdx == 0:
                    W = 0
                    Wp = 0
                elif cIdx == nCols - 1:
                    E = 0
                    Ep = 0
                polarities = (Np, Ep, Sp, Wp)
                edgeTypes = (N, E, S, W)
                signedEdgeTypes = tuple((e * p for e, p in zip(edgeTypes, polarities)))

                # Generate the pieces
                piece = PuzzlePiece(rIdx, cIdx, signedEdgeTypes)
                pieceList.append(piece)

        # self.pieceList = pieceList
        return pieceList

    def generateInitialPuzzleState(self) -> PuzzleState:
        """Generate an initial puzzle state from the basic dimensions of the puzzle.
        This should include anchoring the first corner piece and designating it as
        the first fragment.
        """
        pieceList = self.pieceList
        # Choose the first corner piece as the one to stay in position
        # (we can do this without loss of generality, because any resulting
        # alternate assembled puzzle could be rotated to put this corner in
        # the upper left, and it's orientation in that position cannot have
        # changed either, as long as we are not allowing flipping of pieces!)
        startingCornerPiece = pieceList[0]
        # Anchor the corner piece
        fragmentAnchor = AnchorLocation(
            pieceListIdx=0, anchorGridPosition=(0, 0), anchorOrientation=0
        )
        # startingCornerPiece.newPosition = [0, 0]
        # startingCornerPiece.newOrientation = 0
        fragEdgePairs = FragmentEdgePairSet()
        startingFragment = PuzzleFragment(
            puzzleParameters=self,
            pieceList=[startingCornerPiece],
            fragEdgePairs=fragEdgePairs,
            anchorLocation=fragmentAnchor,
        )
        #
        initPuzzleState = PuzzleState(
            parent=self,
            edgePairs=EdgePairSet(),
            fragments=[startingFragment],
            loosePieces=pieceList[1:],
        )
        return initPuzzleState

    def classifyPiecesAndEdges(
        self,
    ) -> Tuple[
        Dict[int, PuzzlePiece],
        Dict[PuzzlePiece, PieceType],
        Dict[PieceType, List[PuzzlePiece]],
        Dict[EdgeClass, List[int]],
        Dict[int, EdgeClass],
    ]:
        """Build the dicts which allow easy lookups of edge class from edge number,
        edges from edge class, piece from edge number, and piece type from piece.
        """

        edgesFromEdgeClass = {edgeClass: [] for edgeClass in EdgeClass}
        edgeClassFromEdge = dict()
        pieceFromEdgeDict = dict()
        pieceTypeFromPiece = dict()
        for piece in self.pieceList:
            pieceTypeFromPiece[piece] = piece.getPieceType()
            pieceEdgeNums = piece.getEdgeNums()
            pieceEdgeClasses = piece.getEdgeClasses()
            for edgeNum, edgeClass in zip(pieceEdgeNums, pieceEdgeClasses):
                if edgeNum != 0:
                    pieceFromEdgeDict[edgeNum] = piece
                edgeClassFromEdge[edgeNum] = edgeClass
                edgesFromEdgeClass[edgeClass].append(edgeNum)
        # Assemble reverse lookup for pieces of a given PieceType
        piecesFromPieceType = dict()
        for pieceType in PieceType:
            piecesFromPieceType[pieceType] = [
                p for p in self.pieceList if p.pieceType == pieceType
            ]
        return (
            pieceFromEdgeDict,
            pieceTypeFromPiece,
            piecesFromPieceType,
            edgesFromEdgeClass,
            edgeClassFromEdge,
        )

    def getPieceFromEdge(self, edgeNum: int) -> PuzzlePiece:
        return self.pieceFromEdgeDict[edgeNum]


# Consider, do we want to categorize the edges on construction (outer, rim_cw, rim_ccw, internal)
# and put into dict? Or an Edge object class?


# Next task, explore finding the 11 possible edges which could be identified
# with edge #1- (east of upper left corner).  1- is out because it is in
# the original. 2-, 3-,
# Top Row edge pieces West edge
# Right Col edge pieces North edge
# Bottom Row edge pieces East edge
# Left Col edge pieces South edge


# MARK: solve()

listOfErrors = []


def solve(puzzState: PuzzleState, recursionLevel=0) -> PuzzleState:
    print(f"===========\nsolve() recursion level {recursionLevel}:\n===========")
    activeEdge = puzzState.selectActiveEdge()
    print(f"  Active Edge = {activeEdge}")
    candidateEdges = puzzState.findCandidateEdges(activeEdge)
    print(f"  Candidate Edges = {candidateEdges}")
    for candidateEdge in candidateEdges:
        try:
            nextState = puzzState.addConnection((activeEdge, candidateEdge))
            if nextState.isComplete():
                solvedPuzzle = nextState
                return solvedPuzzle
            else:
                solvedPuzzle = solve(nextState, recursionLevel=recursionLevel + 1)
                return solvedPuzzle
        except (AddConnectionError, OutOfEdgesError) as e:
            listOfErrors.append(e)
            continue
    # Ran out of edges without finding a solution!
    raise OutOfEdgesError("Ran out of edges!")


# MARK: Exceptions
class AddConnectionError(Exception):
    pass


class OutOfEdgesError(Exception):
    pass


class InconsistentFragmentCoordinatesError(Exception):
    pass


class GetOrderedEdgesVerificationError(Exception):
    pass


class WrongEdgeCountError(Exception):
    pass


class InvalidPieceError(Exception):
    pass


class UnknownPieceTypeError(Exception):
    pass


class EdgePairError(Exception):
    pass


class LoosePiecesInFragmentError(Exception):
    pass


class FirstFreeEdgeError(Exception):
    pass


class ActiveEdgeNotOnFragmentError(Exception):
    pass


class ZeroEdgeNumberNotUniqueError(Exception):
    pass


def main():
    nRows = 5
    nCols = 5
    puzzGen = PuzzleParameters(nRows, nCols)
    initialPuzzState = puzzGen.initPuzzleState
    pieceList = initialPuzzState.getPieceList()
    for piece in pieceList:
        # print(piece)
        print(f"{piece.origPosition}: {piece.getEdgeNums()}")
    # Try to solve it!
    try:
        solvedPuzzle = solve(initialPuzzState)
    except OutOfEdgesError as e:
        print("Impossible, no solution found!")
        return
    print("Found final solution!!  Figure out a way to print it!")
    solvedPuzzle.show()


if __name__ == "__main__":
    main()
"""
        for rIdx in range(nRows):
            if rIdx==0:
                # Top row, original north edge must be 0.
            elif rIdx==(nRows-1):
                # Bottom row, original south edge must be 0
            else:
                # interior row
            for cIdx in range(nCols):
                if cIdx==0:
                    # Left col, original west edge must be 0
                elif cIdx==(nCols-1):
                    # right col, original east edge must be 0
                else:
                    # interior col
                # Calculate N edge type
                north = (rIdx-1)*(nCols-1)+cIdx
                south = rIdx*(nCols-1)+cIdx
                hOffset = (nRows)(nCols-1)
"""

"""
Some thoughts about complications:

Pieces may be rectangular rather than square. In that case, a long edge
will never connect to a short edge, and rotations can only ever be 180
deg (at least not without allowing major complications like connections
whose corners don't line up!).  On a similar note, real puzzles don't
always have a fixed grid of piece corner locations.  This type of 
complication would make solving this alternate construction problem
incredibly much harder, I think. 

I gave a lot of thought initially to wanting to allow for the 
case where puzzle piece connections could have a neutral polarity;
that is, where a puzzle piece edge could be complimentary to itself.
In such a case, when you choose another edge which is to match it, 
there would actually be 2 possible alternate connection choices rather
than one. In this case, a signed edge assignment doesn't capture all
the possibilities.  But, it makes things more complicated, and for now,
I will just assume this doesn't happen.  It can be guaranteed by making 
sure that all edge connections involve more area for one piece than 
the other, since self connection requires exactly equal area between
the included and excluded regions. 
"""
