# For exploring Parker Alternative Puzzles
# import typing
from typing import Tuple, Optional, List, Dict, NamedTuple, Set
import numpy as np
from enum import Enum

# from collections import namedtuple
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

    def addConnection(self, edge1: int, edge2: int):
        # Validate input
        if edge1 == edge2:
            raise EdgePairError("Not allowed to pair and edge with itself")
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

    def getFlatEdgePairList(self) -> Tuple[int]:
        """Get a flattened list (tuple) of the edge
        numbers involved in all pairs
        """
        return tuple(self._lookup.keys())

    def getDeepCopy(self) -> "EdgePairSet":
        """Return a deep copy of the current EdgePairSet instance"""
        copy = EdgePairSet()
        copy._lookup = {key: value for key, value in self._lookup.items}
        return copy


# MARK: FragmentEdgePairSet
class FragmentEdgePairSet:
    _lookup: Dict[int, int]

    def addConnection(self, edge1: int, edge2: int):
        if edge1 == edge2:
            raise EdgePairError("Not allowed to pair and edge with itself")
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

    def getFlatEdgePairList(self) -> Tuple[int]:
        """Get a flattened list (tuple) of the edge
        numbers involved in all pairs
        """
        return tuple(self._lookup.keys())

    def getDeepCopy(self) -> "EdgePairSet":
        """Return a deep copy of the current EdgePairSet instance"""
        copy = FragmentEdgePairSet()
        copy._lookup = {key: value for key, value in self._lookup.items}
        return copy


# MARK: Enums
class PieceType(Enum):
    CORNER = 2
    BORDER = 1
    INTERIOR = 0


class EdgeClass(Enum):
    OUTER_BORDER = 0
    CW_FROM_CORNER = 1
    CCW_FROM_CORNER = 2
    CW_FROM_BORDER = 3
    CCW_FROM_BORDER = 4
    INTERIOR = 5


partnerEdgeClasses = {
    EdgeClass.CW_FROM_CORNER: (EdgeClass.CCW_FROM_BORDER),
    EdgeClass.CCW_FROM_CORNER: (EdgeClass.CW_FROM_BORDER),
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

    def deepCopy(self):
        return FragmentCoordinate(self.rowCoord, self.colCoord, self.rotationCount)


class FragmentCoordDict(dict):
    def deepCopy(self) -> "FragmentCoordDict"["PuzzlePiece", FragmentCoordinate]:
        return {key: val.deepCopy() for key, val in self.items()}


# MARK: PuzzlePiece
class PuzzlePiece:
    def __init__(
        self,
        origRow: int,
        origCol: int,
        signedEdgeNums: Tuple[int, int, int, int],
    ):
        self.origPosition = (origRow, origCol)
        self.pieceType = self.getPieceType()
        self._signedEdgeNums = signedEdgeNums

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

    def getEdgeClasses(self):
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
                    edgeClasses.append(EdgeClass.INTERIOR)
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
                            pieceType == PieceType.BORDER,
                            "To get here, the piece type should only ever be a border edge piece!",
                        )
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

    def __repr__(self) -> str:
        lines = [
            "PuzzlePiece:",
            f"Original Location: {self.origPosition}",
            f"Edge Types: {self.getEdgeNums()}",
            f"Piece Type: {self.pieceType.name}",
        ]
        return "\n  ".join(lines)


def rotateCoord(r, c, rotationOffset):
    """Rotate row,col coordinates by 90deg clockwise rotationOffset times"""
    rotatedRow, rotatedCol = r, c
    for idx in range(rotationOffset):
        rotatedRow, rotatedCol = (rotatedCol, -rotatedRow)
    return rotatedRow, rotatedCol


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

    def __init__(
        self,
        parentPuzzleState: "PuzzleState",
        pieceList: List[PuzzlePiece],
        fragEdgePairs: Optional[FragmentEdgePairSet] = None,
        anchorLocation: Optional[AnchorLocation] = None,
        fragmentCoordDict: Optional[
            FragmentCoordDict[PuzzlePiece, FragmentCoordinate]
        ] = None,
        _cachedFreeEdgesList: Optional[Tuple[int]] = None,
    ):
        self.parentPuzzleState = parentPuzzleState
        self.anchorLocation = anchorLocation
        self.pieceList = pieceList
        if fragEdgePairs is None:
            # Derive from parent puzzle state
            self.fragEdgePairs = self.calcFragEdgePairs(parentPuzzleState, pieceList)
        else:
            self.fragEdgePairs = fragEdgePairs
            # ^^ Should we verify this is valid?? ^^
        if fragmentCoordDict is None:
            self.fragmentCoordDict = FragmentCoordDict()
            self.assignFragmentCoordinates()
        else:
            # TODO: Validate this is OK first?
            self.fragmentCoordDict = fragmentCoordDict
        self._cachedFreeEdgesList = _cachedFreeEdgesList
        if self._cachedFreeEdgesList is None:
            self.updateCachedFreeEdgesList()

    def calcFragEdgePairs(
        puzzleState: "PuzzleState", pieceList: List[PuzzlePiece]
    ) -> FragmentEdgePairSet:
        """Filter the edge pairs of the puzzle state to include only those
        edges present in the pieces in the piece list.
        """
        statePairedEdges = puzzleState.edgePairs.getFlatEdgePairList()
        # Loop over all piece edges and keep those which are in the state edgePairs
        edgesToInclude = [
            e
            for piece in pieceList
            for e in piece.getEdgeNums()
            if e in statePairedEdges
        ]
        fragEdgePairs = FragmentEdgePairSet()
        for e in edgesToInclude:
            partner = puzzleState.edgePairs[e]
            fragEdgePairs.addConnection(e, partner)
        return fragEdgePairs

    def assignFragmentCoordinates(self):
        """Each piece within the fragment needs to be assigned local
        fragment coordinates.  Start from pieceList[0] and work out
        from there. When complete, all fragment pieces should have
        coordinates assigned.
        """
        startingPiece = self.pieceList[0]
        self.fragmentCoordDict[startingPiece] = FragmentCoordinate(0, 0, 0)
        piecesToCheckNext = set([startingPiece])
        # N, E, S, W = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # offsets = (N, E, S, W)
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
                        partnerPiece = self.parentPuzzleState.getPieceFromEdge(
                            partnerEdgeNum
                        )
                        # Calculate fragment coordinate
                        partnerCoord = self.calcFragmentCoord(
                            piece,
                            edgeNum,
                            self.fragmentCoordDict[piece],
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
                            existingFragCoord = self.fragmentCoordDict[
                                partnerPiece
                            ]  # may throw KeyError
                            # Didn't throw error, continue to check if it matches
                            if existingFragCoord != partnerCoord:
                                raise InconsistentFragmentCoordinatesError(
                                    "Inconsistent! This piece already has fragment coordinates, but the new ones calculated don't agree!"
                                )
                        except KeyError:
                            # No current fragment coordinate, assign it
                            self.fragmentCoordDict[partnerPiece] = partnerCoord
                            # Add piece to the set of pieces to check the edges of next
                            piecesToCheckNext.add(partnerPiece)
                # Finished looping over current list of pieces to check

        # Finished looping, all pieces should have consistent fragment coordinates assigned
        if not self.checkAllFragmentCoordinatesAssigned():
            raise LoosePiecesInFragmentError
        # Otherwise, everything is consistent and all pieces have been assigned local fragment coordinates
        # IF the fragment is anchored, I think we should convert all of these local
        # coordinates to true puzzle coordinates
        if self.isAnchored():
            # Convert local coord to absolute coord
            anchorPiece = self.pieceList[self.anchorLocation.pieceListIdx]
            anchorRow, anchorCol = self.anchorLocation.anchorGridPosition
            anchorRotation = self.anchorLocation.anchorOrientation
            localFragCoord = self.fragmentCoordDict[anchorPiece]
            localRowCoord = localFragCoord.rowCoord
            localColCoord = localFragCoord.colCoord
            localRotation = localFragCoord.rotationCount
            rowOffset = anchorRow - localRowCoord
            colOffset = anchorCol - localColCoord
            rotationOffset = (anchorRotation - localRotation) % 4
            for piece in self.pieceList:
                # Get old fragment coordinate, translate first then apply rotation
                oldFragCoord = self.fragmentCoordDict[piece]
                translatedRowCoord = oldFragCoord.rowCoord + rowOffset
                translatedColCoord = oldFragCoord.colCoord + colOffset
                # Apply rotation
                # if the rotation offset is 1, the whole fragment should rotate clockwise
                # 90 deg around the origin.  So, a piece at (r,c) should end up at (c, -r)
                rotatedRowCoord, rotatedColCoord = rotateCoord(
                    translatedRowCoord, translatedColCoord, rotationOffset
                )
                rotatedRotationCount = oldFragCoord.rotationCount + rotationOffset
                newFragCoord = FragmentCoordinate(
                    rotatedRowCoord,
                    rotatedColCoord,
                    rotatedRotationCount,
                )
                # Replace with updated coordinate (should we update the existing one instead of making a new one?)
                self.fragmentCoordDict[piece] = newFragCoord

    def calcFragmentCoord(
        self,
        anchorPiece: PuzzlePiece,
        anchorEdgeNum: int,
        anchorPieceCoord: FragmentCoordinate,
        partnerPiece: PuzzlePiece,
        partnerEdgeNum: int,
    ):
        """Calculate the relative fragment coordinate for the partner piece,
        based on the location of the anchor piece, its orientation, and the
        linked edges orientations.
        TODO: no need for this to be instance method for PuzzleFragment, OR,
        could drop the anchorPieceCoord input, which can be dervied from self
        and anchorPiece.
        """
        # Determine which way the anchor edge is facing in the fragment
        # coordinate system
        originalOrientation = anchorPiece.getEdgeNums().index(anchorEdgeNum)
        anchorEdgeOrientation = (
            originalOrientation + anchorPieceCoord.rotationCount
        ) % 4
        # The partner edge faces the opposite direction
        partnerEdgeOrientation = (anchorEdgeOrientation + 2) % 4
        # How rotated is this orientation from the original orientation?
        partnerEdgeOriginalOrientation = partnerPiece.getEdgeNums().index(
            partnerEdgeNum
        )
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

    def checkAllFragmentCoordinatesAssigned(self):
        """Return True if all pieces have fragment coordinates
        assigned, return False otherwise.
        """
        assignedFlags = [
            piece in self.fragmentCoordDict.keys() for piece in self.pieceList
        ]
        allAssignedFlag = np.all(assignedFlags)
        return allAssignedFlag

    def getFreeEdgesList(self):
        if self._cachedFreeEdgesList is None:
            self.updateCachedFreeEdgesList()
        return self._cachedFreeEdgesList

    def updateCachedFreeEdgesList(self):
        """Free edges are those which are on pieces in this fragment,
        but which are not in the list of edgePairs
        """
        flatPairedEdgeList = self.fragEdgePairs.getFlatEdgePairList()
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
        self._cachedFreeEdgesList = freeEdges

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
            localRot = self.fragmentCoordDict[curPiece].rotationCount
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
        freeEdges = self.getFreeEdgesList()
        # TODO: I think this can safely be replaced with
        # return self.edgePairs[edgeNum] is None
        return edgeNum in freeEdges

    def getEdgeDirection(self, edgeNum: int) -> int:
        """Get the direction an edge faces in the fragment coordinate grid"""
        if edgeNum == 0:
            raise ZeroEdgeNumberNotUniqueError(
                "Tried to check the edge direction of a non-unique edge number (0)!"
            )
        piece = self.parentPuzzleState.getPieceFromEdge(edgeNum)
        coord = self.fragmentCoordDict[piece]
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
        # The search pattern is different for finding the first edge
        firstFreeEdgeNum, firstFreeEdgePiece, mostRecentEdgeDirection = (
            self.findFirstFreeEdge(startingPiece=self.pieceList[0])
        )
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
                # Edge is free, add it to the list
                externalEdgeList.append((edgeNumToCheck, currentPiece))
                # Keep current piece, update most recent edge direction
                mostRecentEdgeDirection = searchDirection
            else:
                # Edge not free, move to the connected piece in the search
                # direction
                partnerEdge = self.fragEdgePairs[edgeNumToCheck]
                partnerPiece = self.parentPuzzleState.getPieceFromEdge(partnerEdge)
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
                    inEdgePairs = self.frag[edge] is not None
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
        return self.anchorLocation is None

    def deepCopy(self) -> "PuzzleFragment":
        pieceListCopy = [*self.pieceList]
        anchorLocCopy = self.anchorLocation.deepCopy() if self.anchorLocation else None
        coordDictCopy = self.fragmentCoordDict.deepCopy()
        copy = PuzzleFragment(
            self.parentPuzzleParamObj,  # ok to use orig
            pieceList=pieceListCopy,
            edgePairs=self.edgePairs.getDeepCopy(),
            anchorLocation=anchorLocCopy,
            fragmentCoordDict=coordDictCopy,
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
        otherEdgesToPair = tuple(-e for e in edgesToPair)
        edge1 = edgesToPair[0]
        edge2 = edgesToPair[1]
        edge1c = otherEdgesToPair[0]
        edge2c = otherEdgesToPair[1]
        # Check whether a piece self-connects
        if self.getPieceFromEdge(edge1) == self.getPieceFromEdge(edge2):
            raise AddConnectionError(
                "Candidate edge pair leads to piece self connection!"
            )
        if self.getPieceFromEdge(edge1c) == self.getPieceFromEdge(edge2c):
            raise AddConnectionError(
                "Candidate edge pair complements lead to piece self connection!"
            )
        # Check whether a fragment self connects
        frag1 = self.getFragmentFromEdge(edge1)
        if frag1 == self.getFragmentFromEdge(edge2):
            # Both edges are on the same fragment.  This is only allowable
            # if it works out geometrically. We can test this by checking
            # the coordinates of each piece involved and the orientation of
            # each edge from that piece.  If they lie on the same edge location,
            # then this is allowable (and indeed, required), otherwise, this
            # implies the fragment folds in some way to self connect, and is
            # a contradiction.
            if not self.intraFragmentConnnectionOK(frag1, edge1, edge2):
                raise AddConnectionError(
                    "Candidate edge pair leads to non-working fragment self connection!"
                )
        frag1c = self.getFragmentFromEdge(edge1c)
        if frag1c == self.getFragmentFromEdge(edge2c):
            if not self.intraFragmentConnnectionOK(frag1c, edge1c, edge2c):
                raise AddConnectionError(
                    "Candidate edge pair complements lead to non-working fragment self connection"
                )

        # Build the new puzzle state
        newPuzzleState = self.buildNewPuzzleState(edgesToPair)
        # Do geometrical checks on new puzzle state

        # Any additional new edge pairs required by the geometrical arrangement
        # of pieces? If so, we need to test adding them before returning

        return newPuzzleState

    def buildNewPuzzleState(self, edgesToPair: Tuple[int, int]):
        """Build a new puzzle state from the current puzzle state (self) and
        a new pair connection.  It is important to make copies of any objects
        which will be modified from state to state, because recursive calls
        should be able to go back to the prior state without any lingering
        effects.
        """
        newEdgePairs = self.edgePairs.getDeepCopy()
        newEdgePairs.addConnection(*edgesToPair)
        # Build new fragment list
        edge1, edge2 = (*edgesToPair,)
        newFragmentList, newLoosePieces = self.joinEdges(edge1, edge2)

        edge1c, edge2c = (-edge1, -edge2)
        frag1 = self.getFragmentFromEdge()
        self.fragments
        newPuzzleState = PuzzleState(
            self.parent,
            edgePairs=newEdgePairs,
            fragments=newFragmentList,
            loosePieces=newLoosePieces,
        )

    def joinEdges(
        self, edge1, edge2, fragments=None, loosePieces=None
    ) -> Tuple[List[PuzzleFragment], List[PuzzlePiece]]:
        """Return new fragments and loose pieces lists after carrying out
        a connection between the input edges
        """
        if fragments is None:
            fragments = self.fragments
        if loosePieces is None:
            loosePieces = self.loosePieces
        frag1 = self.getFragmentFromEdge(edge1)  # none if on loose piece
        frag2 = self.getFragmentFromEdge(edge2)  # None if on loose piece
        if frag1 == frag2:
            # On same fragment, double-check if it is OK
            if not self.intraFragmentConnnectionOK(frag1, edge1, edge2):
                raise AddConnectionError("Illegal fragment self join in joinEdges!")
            # Otherwise OK to join.  Minimal changes.  Same fragments with the same
            # piece lists, anchor location, fragment coordinate dictionary; just add one
            # edge pair
            newFrag1 = frag1.deepCopy()
            newFrag1.edgePairs.addConnection(edge1, edge2)
            otherFrags = [frag.deepCopy() for frag in fragments if frag != frag1]
            newFragments = [newFrag1].extend(otherFrags)
            # No change in pieceList
            newLoosePieces = list([p for p in loosePieces])
        elif (frag1 is None) and (frag2 is None):
            # Neither edge is on an existing fragment.  Join the two
            # loose pieces to create a new fragment
            piece1 = self.getPieceFromEdge(edge1)
            piece2 = self.getPieceFromEdge(edge2)
            newEdgePairs = self.edgePairs.getDeepCopy()
            newEdgePairs.addConnection(edge1, edge2)
            newFrag = PuzzleFragment(
                self.parent, [piece1, piece2], newEdgePairs, anchorLocation=None
            )
            # TODO: consider whether copying fragments is really necessary or
            # whether unmodified fragments can just be passed along
            newFragments = [frag.deepCopy() for frag in fragments]
            newFragments.append(newFrag)
            newLoosePieces = list([p for p in loosePieces if p not in (piece1, piece2)])
        elif (frag1 is not None) and (frag2 is not None):
            # The two edges are each on different fragments, we need to join the two
            # fragments into a single new fragment
            pass
        else:
            # One is on a fragment and the other is on a loose piece,
            # join the loose piece to the fragment
            if frag1 is None:
                # edge2 is on fragment, edge1 on loose piece
                frag = frag2.deepCopy()
                piece = self.getPieceFromEdge(edge1)
                otherFrags = [fragm.deepCopy() for fragm in fragments if fragm != frag2]
            else:
                # edge1 is on fragment, edge2 on loose piece
                frag = frag1.deepCopy()
                piece = self.getPieceFromEdge(edge2)
                otherFrags = [fragm.deepCopy() for fragm in fragments if fragm != frag1]
            frag.pieceList.append(piece)
            frag.edgePairs

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
            for edge in edgeList:
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
        candidateEdgeClasses = partnerEdgeClasses[activeEdgeClass]
        candidateEdges = []
        for edgeClass in candidateEdgeClasses:
            candidateEdges.extend(self.getEdgesFromEdgeClass(edgeClass))
        # Remove any edges which are already paired
        alreadyPairedEdges = self.edgePairs.getFlatEdgePairList()
        candidateEdges = [
            edge for edge in candidateEdges if edge not in alreadyPairedEdges
        ]
        # Narrow down by piece type (geometrical considerations, especially for
        # anchored fragments)
        allowedPieceTypes = self.getAllowedPieceTypes(activeEdge)
        candidateEdges = [
            edge
            for edge in candidateEdges
            if self.getPieceTypeFromEdge(edge) in allowedPieceTypes
        ]
        # Any other considerations before trying? Should the same checks
        # be run for the implied next edge pair?  Probably a good idea
        candidateEdgesToRemove = []
        for edge in candidateEdges:
            impliedPair = [-edge, -activeEdge]
            edgeClass0 = self.getEdgeClass(impliedPair[0])
            edgeClass1 = self.getEdgeClass(impliedPair[1])
            # Are edge classes compatible?
            if (edgeClass0 not in partnerEdgeClasses[edgeClass1]) or (
                edgeClass1 not in partnerEdgeClasses[edgeClass0]
            ):
                # -activeEdge is not the right edge class to connect to -edge or vice versa
                candidateEdgesToRemove.append(edge)
            elif -edge in self.edgePairs.getFlatEdgePairList():
                # I think this is redundant...
                candidateEdgesToRemove.append(edge)
            elif not allowablePieceType:
                candidateEdgesToRemove.append(edge)

        return tuple(candidateEdges)

    def getPieceTypeFromEdge(self, edgeNum: int) -> PieceType:
        """Get Piece type from edge number"""
        if edgeNum == 0:
            raise ZeroEdgeNumberNotUniqueError(
                "Can't get piece type from zero edge number because it is not unique!"
            )
        piece = self.getPieceFromEdge(edgeNum)
        return self.parent.pieceTypeFromPiece(piece)

    def getAllowedPieceTypes(self, activeEdge) -> Tuple[PieceType]:
        """Based on geometrical considerations, what piece type or types could
        go in the location complementary to the active edge.  This should be
        straightforward to discern if the active edge is on an anchored
        fragment, but might be trickier to narrow down for a floating fragment.
        """
        activePiece = self.getPieceFromEdge(activeEdge)
        activeFragment = self.getFragmentFromEdge(activeEdge)
        if activeFragment is None:
            raise ActiveEdgeNotOnFragmentError(
                "The active edge was not found on a puzzle fragment!"
            )
        lastRow = self.parent.nRows - 1
        lastCol = self.parent.nCols - 1
        if activeFragment.isAnchored():
            activePieceCoord = activeFragment.fragmentCoordDict[activePiece]
            activeEdgeOrigDir = activePiece.getEdgeNums().index[activeEdge]
            activeEdgeDir = activePieceCoord.rotationCount + activeEdgeOrigDir
            # N,E,S,W
            rowOffset = (-1, 0, 1, 0)[activeEdgeDir]
            colOffset = (0, 1, -1, 0)[activeEdgeDir]
            partnerPieceRow = activePieceCoord.rowCoord + rowOffset
            partnerPieceCol = activePieceCoord.colCoord + colOffset
            onHorizBorder = (partnerPieceRow == 0) or (partnerPieceRow == lastRow)
            onVertBorder = (partnerPieceCol == 0) or (partnerPieceCol == lastCol)
            if (not onHorizBorder) and (not onVertBorder):
                allowedPieceTypes = PieceType.INTERIOR
            elif onHorizBorder and onVertBorder:
                allowedPieceTypes = PieceType.CORNER
            else:
                # Not interior or corner, must be on border
                allowedPieceTypes = PieceType.BORDER
        else:
            # TODO: Add conditions narrowing down allowed piece types
            # For now, we are just allowing any piece type on the active
            # edge of a floating fragment.
            allowedPieceTypes = (PieceType.CORNER, PieceType.BORDER, PieceType.INTERIOR)

    def getFragmentFromEdge(self, edgeNum: int) -> Optional[PuzzleFragment]:
        """Find the fragment the given edge is on.  Return None if there is
        no fragment which has this edge (or maybe throw an error?)
        """
        activePiece = self.getPieceFromEdge(edgeNum)
        for fragment in self.fragments:
            if activePiece in fragment.pieceList:
                return fragment
        # Not in any fragment
        return None

    def getEdgeClass(self, edgeNum):
        return self.parent.edgeClassFromEdge[edgeNum]

    def getPieceFromEdge(self, edgeNum):
        return self.parent.pieceFromEdgeDict[edgeNum]

    def getEdgesFromEdgeClass(self, edgeClass):
        return self.parent.edgesFromEdgeClass[edgeClass]

    def getComplementaryEdgeClasses(self, edgeClass):
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
            self.edgesFromEdgeClass,
            self.edgeClassFromEdge,
        ) = self.classifyPiecesAndEdges()
        self.initPuzzleState = self.generateInitialPuzzleState()

    def generatePieces(self) -> List[PuzzlePiece]:
        """Create Puzzle Pieces in initial orientation with initial unique edge numberings
        Edge numbering rules:
        #   Outer flat edges are of type 0 and then vertical edges are numbered in reading
        #   order, then horizontal edges are numbered in reading order.  Polarity is assigned
        #   such that the left or upper side of the edge has polarity +1, while the right or
        #   lower side has polarity -1.  Straight edges are of type 0 and have polarity 0.
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
                signedEdgeTypes = (e * p for e, p in zip(edgeTypes, polarities))

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
        edgePairs = EdgePairSet()
        startingFragment = PuzzleFragment(
            parentPuzzleParamObj=self,
            pieceList=[startingCornerPiece],
            edgePairs=edgePairs,
            anchorLocation=fragmentAnchor,
        )
        #
        initPuzzleState = PuzzleState(
            self.nRows,
            self.nCols,
            edgePairs=edgePairs,
            fragments=[startingFragment],
            loosePieces=pieceList[1:],
        )
        return initPuzzleState

    def classifyPiecesAndEdges(
        self,
    ) -> Tuple[
        Dict[int, PuzzlePiece],
        Dict[PuzzlePiece, PieceType],
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
        return (
            pieceFromEdgeDict,
            pieceTypeFromPiece,
            edgesFromEdgeClass,
            edgeClassFromEdge,
        )


# Consider, do we want to categorize the edges on construction (outer, rim_cw, rim_ccw, internal)
# and put into dict? Or an Edge object class?


# Next task, explore finding the 11 possible edges which could be identified
# with edge #1- (east of upper left corner).  1- is out because it is in
# the original. 2-, 3-,
# Top Row edge pieces West edge
# Right Col edge pieces North edge
# Bottom Row edge pieces East edge
# Left Col edge pieces South edge


def main():
    nRows = 5
    nCols = 5
    puzzGen = PuzzleParameters(nRows, nCols)
    initialPuzzState = puzzGen.initPuzzleState
    pieceList = initialPuzzState.getPieceList()
    for piece in pieceList:
        # print(piece)
        print(f"{piece.origPosition}: {piece.getSignedEdges()}")
    # Try to solve it!
    try:
        solvedPuzzle = solve(initialPuzzState)
    except OutOfEdgesError as e:
        print("Impossible, no solution found!")
        return
    print("Found final solution!!  Figure out a way to print it!")
    solvedPuzzle.show()


listOfErrors = []


def solve(puzzState: PuzzleState) -> PuzzleState:
    activeEdge = puzzState.selectActiveEdge()
    candidateEdges = puzzState.findCandidateEdges(activeEdge)
    for candidateEdge in candidateEdges:
        try:
            nextState = puzzState.addConnection(candidateEdge)
            if nextState.isComplete():
                solvedPuzzle = nextState
                return solvedPuzzle
            else:
                solvedPuzzle = solve(nextState)
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
