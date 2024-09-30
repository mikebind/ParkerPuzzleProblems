# For exploring Parker Alternative Puzzles
import typing
from typing import Tuple, Optional, List, Dict, NamedTuple
import numpy as np
from enum import Enum
from collections import namedtuple
from bidict import bidict


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


class EdgePairError(Exception):
    pass


class LoosePiecesInFragmentError(Exception):
    pass


class PieceType(Enum):
    CORNER = 2
    EDGE = 1
    INTERIOR = 0


# Define a named tuple for anchor location info
class AnchorLocation(NamedTuple):
    pieceListIdx: int
    anchorGridPosition: Tuple[int, int]
    anchorOrientation: int


class FragmentCoordinate(NamedTuple):
    rowCoord: int
    colCoord: int
    rotationCount: int


# MARK: PieceEdge
# TODO: Consider eliminating, I think EdgePairList and edgeToPieceDict
# pretty much do what I want this class to enable, and I think this
# just needlessly complicates things
class PieceEdge:
    def __init__(
        self,
        edgeInt: int,
        parent: "PuzzlePiece",
        oldPartner: "PieceEdge",
        newPartner: Optional["PieceEdge"],
    ):
        self.parent = parent
        self.oldPartner = oldPartner
        self.newPartner = newPartner
        self.edgeNum = edgeInt


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
        self.edges = (
            PieceEdge(edgeNum, self, -edgeNum, newPartner=None)
            for edgeNum in signedEdgeNums
        )

    def getEdgeNums(self):
        return tuple(edge.edgeNum for edge in self.edges)

    def getPieceType(self) -> PieceType:
        """Determine piece type by kind and location of straight edges."""
        edges = self.edges
        if len(edges) != 4:
            raise Exception(
                f"Pieces are expected to have exactly 4 edges, but this one has {len(edges)} edges!"
            )
        edgeNums = [edge.edgeNum for edge in self.edges]
        numStraightEdges = np.sum(np.array(edgeNums) == 0)
        if numStraightEdges == 2:
            # The straight edges must be adjacent
            if (edgeNums[0] == 0 and edgeNums[2] == 0) or (
                edgeNums[1] == 0 and edgeNums[3] == 0
            ):
                raise Exception(
                    f"Invalid piece with straight edges on non-adjacent sides."
                )
            pieceType = PieceType.CORNER
        elif numStraightEdges == 1:
            pieceType = PieceType.EDGE
        elif numStraightEdges == 0:
            pieceType = PieceType.INTERIOR
        else:
            raise Exception(f"Unknown piece type with {numStraightEdges} edges.")
        return pieceType

    def getCWEdges(self, startIdx) -> Tuple[Tuple[int], Tuple[int]]:
        """Get the clockwise sequence of edges, starting from startIdx"""
        cwEdges = (*self.edges[startIdx:], *self.edges[0:startIdx])
        return cwEdges

    def __repr__(self) -> str:
        lines = [
            "PuzzlePiece:",
            f"Original Location: {self.origPosition}",
            f"Edge Types: {self.getEdgeNums()}",
            f"Piece Type: {self.pieceType.name}",
        ]
        return "\n  ".join(lines)


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

    connectionList = Nx2 array of edges which are connected among the pieces. Each piece
    has a unique set of edges, so just knowing the edge number and sign is enough.
    """

    def __init__(
        self,
        parentPuzzleParamObj: "PuzzleParameters",  # quoted because forward reference
        pieceList: List[PuzzlePiece],
        edgePairs: EdgePairSet,
        anchorLocation: Optional[AnchorLocation] = None,
        fragmentCoordDict: Optional[Dict[PuzzlePiece, FragmentCoordinate]] = None,
        _cachedFreeEdgesList: Optional[Tuple[int]] = None,
    ):
        self.parentPuzzleParamObj = parentPuzzleParamObj
        self.anchorLocation = anchorLocation
        self.pieceList = pieceList
        self.edgePairs = edgePairs
        if fragmentCoordDict is None:
            self.fragmentCoordDict = dict()
            self.fragmentCoordDict = self.assignFragmentCoordinates()
        else:
            # TODO: Validate this is OK first?
            self.fragmentCoordDict = fragmentCoordDict
        self._cachedFreeEdgesList = self.updateCachedFreeEdgesList()

    def assignFragmentCoordinates(self):
        """Each piece within the fragment needs to be assigned local
        fragment coordinates.  Start from pieceList[0] and work out
        from there. When complete, all fragmetn
        """
        startingPiece = self.pieceList[0]
        self.fragmentCoordDict[startingPiece] = (0, 0, 0)
        piecesToCheckNext = set([startingPiece])
        N, E, S, W = ((-1, 0), (0, 1), (1, 0), (0, -1))
        offsets = (N, E, S, W)
        while len(piecesToCheckNext) > 0:
            piecesToCheckNow = piecesToCheckNext
            piecesToCheckNext = set()
            for piece in piecesToCheckNow:
                # check each edge and assign coords
                for edge in piece.edges:
                    # if edgeNum has a partner, assign the corresponding
                    # fragment coordinate to that piece associated with
                    # partner
                    edgeNum = edge.edgeNum
                    partnerEdgeNum = self.edgePairs[edgeNum]
                    if partnerEdgeNum:
                        # partner edge exists, get piece
                        partnerPiece = self.parentPuzzleParamObj.getPieceFromEdgeNum(
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
                            self.fragmentCoordDict[partnerPiece]
                            # Add piece to the set of pieces to check the edges of next
                            piecesToCheckNext.add(partnerPiece)
                # Finished looping over current list of pieces to check

        # Finished looping, all pieces should have consistent fragment coordinates assigned
        if not self.checkAllFragmentCoordinatesAssigned():
            raise LoosePiecesInFragmentError
        # Otherwise, everything is consistent and all pieces have been assigned local fragment coordinates

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
        TODO: no need for this to be instance method for PuzzleFragment
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
        flatPairedEdgeList = self.edgePairs.getFlatEdgePairList()
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

    def findFirstFreeEdge(self) -> Tuple[int, PuzzlePiece, int]:
        """Starting from north on the first piece in the pieceList,
        find the first edge which does not have a connected piece.
        If all edges are connected, then move to the next piece to the
        north and repeat.  Repeat until an edge is found, and then
        return a tuple of the free edge number, the puzzle piece it
        was found on, and the direction it was on that piece (in the
        local fragment coordinate system).
        """
        curPiece = self.pieceList[0]
        localRot = self.fragmentCoordDict[curPiece].rotationCount
        edgeNums = curPiece.getCWEdges(localRot)
        freeEdges = self.getFreeEdgesList()
        for edgeNum in edgeNums:
            self.checkEdgeFree(edgeNum)
        pass

    def checkEdgeFree(self, edgeNum):
        freeEdges = self.getFreeEdgesList()
        # TODO: I think this can safely be replaced with
        # return self.edgePairs[edgeNum] is None
        return edgeNum in freeEdges

    def getOrderedExternalEdgeList(self, verifyFlag: bool = True):
        """Identify a starting free (unconnected) edge.  Then, follow the open
        edges clockwise around the entire fragment. If verifyFlag, then
        do the extra check to make sure all edges are either in the list of
        ordered external edges, or in the edgePairs.  (True for now,
        may change to false later to speed computation after some testing)
        """
        # The search pattern is different for finding the first edge
        firstFreeEdgeNum, firstFreeEdgePiece, mostRecentEdgeDirection = (
            self.findFirstFreeEdge()
        )
        # Start the list of external edges
        externalEdgeList = [(firstFreeEdgeNum, firstFreeEdgePiece)]
        # Need edge number AND piece because 0 edge numbers are not unique mappings
        # and we want to be able to tell whether we have completed the circuit
        """ Once the first free edge is found, store it in the externalEdgeList.
        Then, check the next edge clockwise around the current piece. 
        If a piece is there, move to that piece and check the same edge direction
        as the most recent external edge.  Either there is a piece there or it is
        open.  If open, add it to the list and check the next clockwise edge
        """
        # Now search for the next edge
        currentPiece = firstFreeEdgePiece
        # The search direction is one step clockwise around the current piece
        searchDirection = (mostRecentEdgeDirection + 1) % 4
        while not self.externalEdgeListClosedFlag(externalEdgeList):
            edgeNumToCheck = currentPiece.getEdgeNums()[searchDirection]
            if self.checkEdgeFree(edgeNumToCheck):
                # Edge is free, add it to the list
                externalEdgeList.append((edgeNumToCheck, currentPiece))
                # Keep current piece, update most recent edge direction
                mostRecentEdgeDirection = searchDirection
            else:
                # Edge not free, move to the connected piece in the search
                # direction
                partnerEdge = self.edgePairs[edgeNumToCheck]
                partnerPiece = self.parentPuzzleParamObj.pieceFromEdgeDict[partnerEdge]
                currentPiece = partnerPiece
                # Update the search direction ccw one step
                searchDirection = (searchDirection - 1) % 4
        # External edge list is now a complete closed loop
        # TODO: Verify?? Are all edges accounted for?
        if verifyFlag:
            for piece in self.pieceList:
                for edge in piece.getEdgeNums():
                    inFreeEdges = (edge, piece) in externalEdgeList
                    inEdgePairs = self.edgePairs[edge] is not None
                    if not inFreeEdges and not inEdgePairs:
                        raise GetOrderedEdgesVerificationError(
                            f"Edge {edge} is in neither the external nor the paired edge lists!"
                        )

        return externalEdgeList[:-1]  # drop the repeated edge before returning

    def externalEdgeListClosedFlag(
        self, orderedEdgeList: List[Tuple[int, PuzzlePiece]]
    ) -> bool:
        """Check that the list makes a closed loop"""
        longEnough = len(orderedEdgeList) > 1
        endsMatch = orderedEdgeList[0] == orderedEdgeList[-1]
        return longEnough and endsMatch

    def isValid(self):
        """Check fragment validity, return True or False. Valid if it fits in the current
        parentPuzzleGrid.  This can depend on whether it is anchored or not (for example,
        a horziontal strip might be known to be invalid even if it would fit vertically)
        """
        if self.isAnchored():
            # just check if any puzzle piece in the fragment is located outside the grid
            # Actually, there are more ways to be invalid... wrong piece type connected
            # for example?
            pass


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


# MARK: PuzzleState
class PuzzleState:
    def __init__(
        self,
        nRows: int,
        nCols: int,
        pairedEdgesDict: Dict,
        unpairedEdgesList: List,
        fragments: List[PuzzleFragment],
        loosePieces: List[PuzzlePiece],
    ):
        self.nRows = nRows
        self.nCols = nCols
        self.pairedEdgesDict = pairedEdgesDict
        self.unpairedEdgesList = unpairedEdgesList
        self.fragments = fragments
        self.loosePieces = loosePieces

    def addConnection(self, edgesToPair: Tuple[int, int]) -> "PuzzleState":
        """Starting from the current state, try to add the edge pair connection
        given. If invalid for any reason, throw an AddConnectionError exception.
        If valid, return a new PuzzleState which incorporates the new connection.
        Each new connection involves linking either a fragment to another fragment,
        a fragment to a loose piece, or two loose pieces together.
        Through all manipulations here, we need to make sure not to alter any of the
        present puzzle state's variables, because if there is a problem in this or
        any deeper connection, we need to be able to restore the last state where
        we hadn't found a contradiction.
        """
        otherEdgesToPair = (-e for e in edgesToPair)

        AddConnectionError

        return newPuzzleState

    def selectActiveEdge(self) -> int:
        """Select the next active edge to connect to
        This should be a signed integer.
        """
        self.fragments[0].pieceList[-1]
        return 1

    def findCandidateEdges(self, activeEdge) -> Tuple[int]:

        return tuple(edgeCandidates)

    def getPieceList(self) -> Tuple[PuzzlePiece]:
        """ """
        return pieceList


# MARK: PuzzParameters
class PuzzleParameters:
    def __init__(self, nRows: int, nCols: Optional[int] = None):
        if nCols is None:
            nCols = nRows
        self.nRows = nRows
        self.nCols = nCols
        self.pieceList = self.generatePieces()
        self.initPuzzleState = self.generateInitialPuzzleState()
        # Build handy lookup table for pieces from edge numbers
        pieceFromEdgeDict = dict()
        for piece in self.pieceList:
            for edgeNum in piece.getEdgeNums():
                if edgeNum != 0:
                    pieceFromEdgeDict[edgeNum] = piece
        self.pieceFromEdgeDict = pieceFromEdgeDict

    def generatePieces(self) -> List[PuzzlePiece]:
        # Create Puzzle Pieces in initial orientation with initial unique edge numberings
        # Edge numbering rules:
        #   Outer flat edges are of type 0 and then vertical edges are numbered in reading
        #   order, then horizontal edges are numbered in reading order.  Polarity is assigned
        #   such that the left or upper side of the edge has polarity +1, while the right or
        #   lower side has polarity -1.  Straight edges are of type 0 and have polarity 0.
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
                p = PuzzlePiece(rIdx, cIdx, signedEdgeTypes)
                pieceList.append(p)

        # self.pieceList = pieceList
        return pieceList

    def generateInitialPuzzleState(self) -> PuzzleState:
        """Generate an initial puzzle state from the basic dimensions of the puzzle.
        This should include anchoring the first corner piece and designating it as
        the first fragment.
        """

        pieceList = self.pieceList
        # Make lookup dictionary to go from signed edge to the piece it is on
        pieceFromEdgeDict = dict()
        unpairedEdgesList = []
        for piece in pieceList:
            signedEdges = piece.getSignedEdges()
            for edge in signedEdges:
                unpairedEdgesList.append(edge)
                pieceFromEdgeDict[edge] = piece

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
        startingCornerPiece.newPosition = [0, 0]
        startingCornerPiece.newOrientation = 0
        startingFragment = PuzzleFragment(
            pieceList=[startingCornerPiece],
            connectionList=bidict(),
            anchorLocation=fragmentAnchor,
        )
        #
        initPuzzleState = PuzzleState(
            self.nRows,
            self.nCols,
            pairedEdgesDict=dict(),
            unpairedEdgesList=unpairedEdgesList,
            fragments=[startingFragment],
            loosePieces=pieceList[1:],
        )
        return initPuzzleState


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
