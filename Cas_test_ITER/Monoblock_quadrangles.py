#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.3.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'D:/Recherche/FESTIM/2019 Cas ITER/Maillage')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS


geompy = geomBuilder.New()

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
I = geompy.MakeVertex(0, 0.006, 0)
G = geompy.MakeVertex(0, 0.0075, 0)
F = geompy.MakeVertex(0, 0.008500000000000001, 0)
Q = geompy.MakeRotation(I, OZ, 180*math.pi/180.0)
P = geompy.MakeRotation(G, OZ, 180*math.pi/180.0)
OO = geompy.MakeRotation(F, OZ, 180*math.pi/180.0)
B = geompy.MakeVertex(0, 0.0145, 0)
geomObj_1 = geompy.MakeMarker(0, 0, 0, 1, 0, 0, 0, 1, 0)
J = geompy.MakeRotation(I, OZ, 43.99999999999999*math.pi/180.0)
H = geompy.MakeRotation(G, OZ, 43.99999999999999*math.pi/180.0)
E = geompy.MakeRotation(F, OZ, 43.99999999999999*math.pi/180.0)
K = geompy.MakeRotation(I, OZ, 136*math.pi/180.0)
L = geompy.MakeRotation(G, OZ, 136*math.pi/180.0)
M = geompy.MakeRotation(F, OZ, 136*math.pi/180.0)
C = geompy.MakeVertex(0, 0.0135, 0)
Bp = geompy.MakeVertex(0, 0.01449, 0)
Ap = geompy.MakeVertex(-0.014, 0.01449, 0)
A = geompy.MakeVertex(-0.014, 0.0145, 0)
D = geompy.MakeVertex(-0.014, 0.0135, 0)
N = geompy.MakeVertex(-0.014, -0.0135, 0)
R = geompy.MakeVertex(0, -0.0135, 0)
Ligne_1 = geompy.MakeLineTwoPnt(A, B)
Ligne_2 = geompy.MakeLineTwoPnt(B, Bp)
Ligne_3 = geompy.MakeLineTwoPnt(Bp, Ap)
Ligne_4 = geompy.MakeLineTwoPnt(Ap, A)
Face_1 = geompy.MakeFaceWires([Ligne_1, Ligne_2, Ligne_3, Ligne_4], 1)
Ligne_5 = geompy.MakeLineTwoPnt(Ap, D)
Ligne_6 = geompy.MakeLineTwoPnt(D, C)
Ligne_7 = geompy.MakeLineTwoPnt(C, Bp)
Face_2 = geompy.MakeFaceWires([Ligne_3, Ligne_5, Ligne_6, Ligne_7], 1)
Ligne_8 = geompy.MakeLineTwoPnt(D, E)
Ligne_9 = geompy.MakeArcCenter(O, E, F,False)
Ligne_10 = geompy.MakeLineTwoPnt(F, C)
Face_3 = geompy.MakeFaceWires([Ligne_6, Ligne_8, Ligne_9, Ligne_10], 1)
Ligne_11 = geompy.MakeArcCenter(O, E, M,False)
Ligne_12 = geompy.MakeLineTwoPnt(M, N)
Ligne_13 = geompy.MakeLineTwoPnt(N, D)
Face_4 = geompy.MakeFaceWires([Ligne_8, Ligne_11, Ligne_12, Ligne_13], 1)
Ligne_14 = geompy.MakeLineTwoPnt(N, R)
Ligne_15 = geompy.MakeLineTwoPnt(R, OO)
Ligne_16 = geompy.MakeArcCenter(O, M, OO,False)
Face_5 = geompy.MakeFaceWires([Ligne_12, Ligne_14, Ligne_15, Ligne_16], 1)
Ligne_17 = geompy.MakeLineTwoPnt(OO, P)
Ligne_18 = geompy.MakeArcCenter(O, P, L,False)
Ligne_19 = geompy.MakeLineTwoPnt(L, M)
Face_6 = geompy.MakeFaceWires([Ligne_16, Ligne_17, Ligne_18, Ligne_19], 1)
Ligne_20 = geompy.MakeArcCenter(O, L, H,False)
Ligne_21 = geompy.MakeLineTwoPnt(H, E)
Face_7 = geompy.MakeFaceWires([Ligne_11, Ligne_19, Ligne_20, Ligne_21], 1)
Ligne_22 = geompy.MakeArcCenter(O, H, G,False)
Ligne_23 = geompy.MakeLineTwoPnt(G, F)
Face_8 = geompy.MakeFaceWires([Ligne_9, Ligne_21, Ligne_22, Ligne_23], 1)
Ligne_24 = geompy.MakeLineTwoPnt(G, I)
Ligne_25 = geompy.MakeArcCenter(O, I, J,False)
Ligne_26 = geompy.MakeLineTwoPnt(J, H)
Face_9 = geompy.MakeFaceWires([Ligne_22, Ligne_24, Ligne_25, Ligne_26], 1)
Ligne_27 = geompy.MakeArcCenter(O, J, K,False)
Ligne_28 = geompy.MakeLineTwoPnt(K, L)
Face_10 = geompy.MakeFaceWires([Ligne_20, Ligne_26, Ligne_27, Ligne_28], 1)
Ligne_29 = geompy.MakeArcCenter(O, K, Q,False)
Ligne_30 = geompy.MakeLineTwoPnt(Q, P)
Face_11 = geompy.MakeFaceWires([Ligne_18, Ligne_28, Ligne_29, Ligne_30], 1)
Assemblage_1 = geompy.MakeCompound([Face_1, Face_2, Face_3, Face_4, Face_5, Face_6, Face_7, Face_8, Face_9, Face_10, Face_11])
[Ar__te_1,Ar__te_2,Ar__te_3,Ar__te_4,Ar__te_5,Ar__te_6,Ar__te_7,Ar__te_8,Ar__te_9,Ar__te_10,Ar__te_11,Ar__te_12,Ar__te_13,Ar__te_14,Ar__te_15,Ar__te_16,Ar__te_17,Ar__te_18,Ar__te_19,Ar__te_20,Ar__te_21,Ar__te_22,Ar__te_23,Ar__te_24,Ar__te_25,Ar__te_26,Ar__te_27,Ar__te_28,Ar__te_29,Ar__te_30,Ar__te_31,Ar__te_32,Ar__te_33,Ar__te_34,Ar__te_35,Ar__te_36,Ar__te_37,Ar__te_38,Ar__te_39,Ar__te_40,Ar__te_41,Ar__te_42,Ar__te_43,Ar__te_44] = geompy.ExtractShapes(Assemblage_1, geompy.ShapeType["EDGE"], True)
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( G, 'G' )
geompy.addToStudy( I, 'I' )
geompy.addToStudy( F, 'F' )
geompy.addToStudy( L, 'L' )
geompy.addToStudy( C, 'C' )
geompy.addToStudy( J, 'J' )
geompy.addToStudy( H, 'H' )
geompy.addToStudy( E, 'E' )
geompy.addToStudy( K, 'K' )
geompy.addToStudy( Q, 'Q' )
geompy.addToStudy( P, 'P' )
geompy.addToStudy( OO, 'OO' )
geompy.addToStudy( D, 'D' )
geompy.addToStudy( B, 'B' )
geompy.addToStudy( M, 'M' )
geompy.addToStudy( N, 'N' )
geompy.addToStudy( A, 'A' )
geompy.addToStudy( Ap, 'Ap' )
geompy.addToStudy( Bp, 'Bp' )
geompy.addToStudy( R, 'R' )
geompy.addToStudy( Ligne_1, 'Ligne_1' )
geompy.addToStudy( Ligne_2, 'Ligne_2' )
geompy.addToStudy( Ligne_3, 'Ligne_3' )
geompy.addToStudy( Ligne_4, 'Ligne_4' )
geompy.addToStudy( Face_1, 'Face_1' )
geompy.addToStudy( Ligne_5, 'Ligne_5' )
geompy.addToStudy( Ligne_6, 'Ligne_6' )
geompy.addToStudy( Ligne_7, 'Ligne_7' )
geompy.addToStudy( Face_2, 'Face_2' )
geompy.addToStudy( Ligne_8, 'Ligne_8' )
geompy.addToStudy( Ligne_9, 'Ligne_9' )
geompy.addToStudy( Ligne_10, 'Ligne_10' )
geompy.addToStudy( Face_3, 'Face_3' )
geompy.addToStudy( Ligne_11, 'Ligne_11' )
geompy.addToStudy( Ligne_12, 'Ligne_12' )
geompy.addToStudy( Ligne_13, 'Ligne_13' )
geompy.addToStudy( Face_4, 'Face_4' )
geompy.addToStudy( Ligne_14, 'Ligne_14' )
geompy.addToStudy( Ligne_15, 'Ligne_15' )
geompy.addToStudy( Ligne_16, 'Ligne_16' )
geompy.addToStudy( Face_5, 'Face_5' )
geompy.addToStudy( Ligne_17, 'Ligne_17' )
geompy.addToStudy( Ligne_18, 'Ligne_18' )
geompy.addToStudy( Ligne_19, 'Ligne_19' )
geompy.addToStudy( Face_6, 'Face_6' )
geompy.addToStudy( Ligne_20, 'Ligne_20' )
geompy.addToStudy( Ligne_21, 'Ligne_21' )
geompy.addToStudy( Face_7, 'Face_7' )
geompy.addToStudy( Ligne_22, 'Ligne_22' )
geompy.addToStudy( Ligne_23, 'Ligne_23' )
geompy.addToStudy( Face_8, 'Face_8' )
geompy.addToStudy( Ligne_24, 'Ligne_24' )
geompy.addToStudy( Ligne_25, 'Ligne_25' )
geompy.addToStudy( Ligne_26, 'Ligne_26' )
geompy.addToStudy( Face_9, 'Face_9' )
geompy.addToStudy( Ligne_27, 'Ligne_27' )
geompy.addToStudy( Ligne_28, 'Ligne_28' )
geompy.addToStudy( Face_10, 'Face_10' )
geompy.addToStudy( Ligne_29, 'Ligne_29' )
geompy.addToStudy( Ligne_30, 'Ligne_30' )
geompy.addToStudy( Face_11, 'Face_11' )
geompy.addToStudy( Assemblage_1, 'Assemblage_1' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_1, 'Arête_1' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_2, 'Arête_2' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_3, 'Arête_3' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_4, 'Arête_4' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_5, 'Arête_5' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_6, 'Arête_6' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_7, 'Arête_7' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_8, 'Arête_8' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_9, 'Arête_9' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_10, 'Arête_10' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_11, 'Arête_11' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_12, 'Arête_12' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_13, 'Arête_13' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_14, 'Arête_14' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_15, 'Arête_15' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_16, 'Arête_16' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_17, 'Arête_17' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_18, 'Arête_18' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_19, 'Arête_19' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_20, 'Arête_20' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_21, 'Arête_21' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_22, 'Arête_22' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_23, 'Arête_23' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_24, 'Arête_24' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_25, 'Arête_25' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_26, 'Arête_26' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_27, 'Arête_27' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_28, 'Arête_28' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_29, 'Arête_29' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_30, 'Arête_30' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_31, 'Arête_31' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_32, 'Arête_32' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_33, 'Arête_33' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_34, 'Arête_34' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_35, 'Arête_35' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_36, 'Arête_36' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_37, 'Arête_37' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_38, 'Arête_38' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_39, 'Arête_39' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_40, 'Arête_40' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_41, 'Arête_41' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_42, 'Arête_42' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_43, 'Arête_43' )
geompy.addToStudyInFather( Assemblage_1, Ar__te_44, 'Arête_44' )
[Face_1_1, Face_2_1, Face_3_1, Face_4_1, Face_5_1, Face_6_1, Face_7_1, Face_8_1, Face_9_1, Face_10_1, Face_11_1] = geompy.RestoreGivenSubShapes(Assemblage_1, [Face_1, Face_2, Face_3, Face_4, Face_5, Face_6, Face_7, Face_8, Face_9, Face_10, Face_11], GEOM.FSM_GetInPlace, False, False)
Auto_group_for_Sous_maillage_1 = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["FACE"])
geompy.UnionList(Auto_group_for_Sous_maillage_1, [Face_1_1, Face_2_1, Face_3_1, Face_4_1, Face_5_1, Face_6_1, Face_7_1, Face_8_1, Face_9_1, Face_10_1, Face_11_1])
Auto_group_for_Bords_cercles = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_Bords_cercles, [Ar__te_13, Ar__te_14, Ar__te_21, Ar__te_22, Ar__te_23, Ar__te_24, Ar__te_25, Ar__te_26, Ar__te_38, Ar__te_39, Ar__te_40, Ar__te_41])
Auto_group_for_Sous_maillage_1_1 = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_Sous_maillage_1_1, [Ar__te_27, Ar__te_28, Ar__te_29, Ar__te_30, Ar__te_31, Ar__te_32, Ar__te_33, Ar__te_34, Ar__te_35, Ar__te_36])
Auto_group_for_Sous_maillage_1_2 = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_Sous_maillage_1_2, [Ar__te_13, Ar__te_14, Ar__te_21, Ar__te_22, Ar__te_23, Ar__te_24, Ar__te_25, Ar__te_26, Ar__te_38, Ar__te_39, Ar__te_40, Ar__te_41])
Auto_group_for_cercle_middle = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_cercle_middle, [Ar__te_1, Ar__te_9, Ar__te_10, Ar__te_11, Ar__te_12, Ar__te_20])
Auto_group_for_bords_rectangles_bas = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_bords_rectangles_bas, [Ar__te_4, Ar__te_5, Ar__te_6, Ar__te_7, Ar__te_37, Ar__te_42])
Auto_group_for_bords_horizontaux = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_bords_horizontaux, [Ar__te_8, Ar__te_15, Ar__te_16, Ar__te_17, Ar__te_18, Ar__te_19])
Auto_group_for_bord_43 = geompy.CreateGroup(Assemblage_1, geompy.ShapeType["EDGE"])
geompy.UnionList(Auto_group_for_bord_43, [Ar__te_2, Ar__te_43])
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_Sous_maillage_1, 'Auto_group_for_Sous-maillage_1' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_Bords_cercles, 'Auto_group_for_Bords_cercles' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_Sous_maillage_1_1, 'Auto_group_for_Sous-maillage_1' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_Sous_maillage_1_2, 'Auto_group_for_Sous-maillage_1' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_cercle_middle, 'Auto_group_for_cercle_middle' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_bords_rectangles_bas, 'Auto_group_for_bords_rectangles_bas' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_bords_horizontaux, 'Auto_group_for_bords_horizontaux' )
geompy.addToStudyInFather( Assemblage_1, Auto_group_for_bord_43, 'Auto_group_for_bord_43' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

Regular_1D = smesh.CreateHypothesis('Regular_1D')
Quadrangle_2D = smesh.CreateHypothesis('Quadrangle_2D')
Number_of_Segments_bords_cercles = smesh.CreateHypothesis('NumberOfSegments')
Number_of_Segments_bords_cercles.SetNumberOfSegments( 15 )
Number_of_Segments_arc_cerlces_inf_sup = smesh.CreateHypothesis('NumberOfSegments')
Maillage_1 = smesh.Mesh(Assemblage_1)
status = Maillage_1.AddHypothesis(Regular_1D,Auto_group_for_Sous_maillage_1_1)
status = Maillage_1.AddHypothesis(Number_of_Segments_arc_cerlces_inf_sup,Auto_group_for_Sous_maillage_1_1)
status = Maillage_1.AddHypothesis(Regular_1D,Auto_group_for_Sous_maillage_1_2)
status = Maillage_1.AddHypothesis(Number_of_Segments_bords_cercles,Auto_group_for_Sous_maillage_1_2)
Number_of_Segments_arc_middle = smesh.CreateHypothesis('NumberOfSegments')
status = Maillage_1.AddHypothesis(Regular_1D,Auto_group_for_cercle_middle)
status = Maillage_1.AddHypothesis(Number_of_Segments_arc_middle,Auto_group_for_cercle_middle)
Number_of_Segments_rectanglebas = smesh.CreateHypothesis('NumberOfSegments')
Number_of_Segments_rectanglebas.SetNumberOfSegments( 50 )
status = Maillage_1.AddHypothesis(Regular_1D,Auto_group_for_bords_rectangles_bas)
status = Maillage_1.AddHypothesis(Number_of_Segments_rectanglebas,Auto_group_for_bords_rectangles_bas)
status = Maillage_1.AddHypothesis(Regular_1D,Auto_group_for_bords_horizontaux)
status = Maillage_1.AddHypothesis(Number_of_Segments_arc_cerlces_inf_sup,Auto_group_for_bords_horizontaux)
bord_43 = smesh.CreateHypothesis('NumberOfSegments')
status = Maillage_1.AddHypothesis(Regular_1D,Auto_group_for_bord_43)
status = Maillage_1.AddHypothesis(bord_43,Auto_group_for_bord_43)
bord_43.SetExpressionFunction( '7*t' )
Number_of_Segments_bords_cercles.SetNumberOfSegments( 15 )
Number_of_Segments_arc_cerlces_inf_sup.SetNumberOfSegments( 50 )
#Maillage_1.GetMesh().RemoveSubMesh( smeshObj_1 ) ### smeshObj_1 has not been yet created
status = Maillage_1.AddHypothesis(Regular_1D,Ar__te_43)
status = Maillage_1.AddHypothesis(bord_43,Ar__te_43)
bord_2 = smesh.CreateHypothesis('NumberOfSegments')
status = Maillage_1.AddHypothesis(Regular_1D,Ar__te_2)
status = Maillage_1.AddHypothesis(bord_2,Ar__te_2)
bord_2.SetNumberOfSegments( 100 )
bord_2.SetConversionMode( 1 )
bord_2.SetReversedEdges( [] )
bord_2.SetObjectEntry( "0:1:1:84" )
bord_2.SetTableFunction( [ 0, 1, 1, 0.01 ] )
bord_43.SetNumberOfSegments( 100 )
bord_43.SetConversionMode( 1 )
bord_43.SetReversedEdges( [] )
bord_43.SetObjectEntry( "0:1:1:84" )
bord_43.SetTableFunction( [ 0, 0.01, 1, 1 ] )
bord_44 = smesh.CreateHypothesis('NumberOfSegments')
status = Maillage_1.AddHypothesis(Regular_1D,Ar__te_44)
status = Maillage_1.AddHypothesis(bord_44,Ar__te_44)
bord_3 = smesh.CreateHypothesis('NumberOfSegments')
status = Maillage_1.AddHypothesis(Regular_1D,Ar__te_3)
status = Maillage_1.AddHypothesis(bord_3,Ar__te_3)
status = Maillage_1.AddHypothesis(Quadrangle_2D)
Number_of_Segments_arc_middle.SetNumberOfSegments( 90 )
bord_44.SetNumberOfSegments( 500 )
bord_44.SetConversionMode( 1 )
bord_44.SetReversedEdges( [] )
bord_44.SetObjectEntry( "0:1:1:84" )
bord_44.SetTableFunction( [ 0, 1, 0.15, 0.02, 1, 0.001 ] )
bord_3.SetNumberOfSegments( 500 )
bord_3.SetConversionMode( 1 )
bord_3.SetReversedEdges( [] )
bord_3.SetObjectEntry( "0:1:1:84" )
bord_3.SetTableFunction( [ 0, 0.001, 0.85, 0.02, 1, 1 ] )
smesh.SetName(Maillage_1, 'Maillage_1')
try:
  Maillage_1.ExportMED(r'D:/Recherche/FESTIM/2019 Cas ITER/Maillage/Maillage_1.med',auto_groups=0,minor=40,overwrite=1,meshPart=None,autoDimension=1)
  pass
except:
  print('ExportMED() failed. Invalid file name?')
try:
  Maillage_1.ExportUNV( r'D:/Recherche/FESTIM/2019 Cas ITER/Maillage/Maillage_1.unv' )
  pass
except:
  print('ExportUNV() failed. Invalid file name?')
try:
  Maillage_1.ExportDAT( r'D:/Recherche/FESTIM/2019 Cas ITER/Maillage/Maillage_1.dat' )
  pass
except:
  print('ExportDAT() failed. Invalid file name?')
try:
  Maillage_1.ExportGMF(r'D:/Recherche/FESTIM/2019 Cas ITER/Maillage/Maillage_1.mesh',Maillage_1)
  pass
except:
  print('ExportGMF() failed. Invalid file name?')
isDone = Maillage_1.Compute()
cercles_haut_bas = Maillage_1.GetSubMesh( Auto_group_for_Sous_maillage_1_1, 'Sous-maillage_1' )
bords_cercles = Maillage_1.GetSubMesh( Auto_group_for_Sous_maillage_1_2, 'Sous-maillage_1' )
cercle_middle = Maillage_1.GetSubMesh( Auto_group_for_cercle_middle, 'cercle_middle' )
bords_rectangles_bas = Maillage_1.GetSubMesh( Auto_group_for_bords_rectangles_bas, 'bords_rectangles_bas' )
bords_horizontaux = Maillage_1.GetSubMesh( Auto_group_for_bords_horizontaux, 'bords_horizontaux' )
bord43 = Maillage_1.GetSubMesh( Ar__te_43, 'bord43' )
bord2 = Maillage_1.GetSubMesh( Ar__te_2, 'bord2' )
bord44 = Maillage_1.GetSubMesh( Ar__te_44, 'bord44' )
bord3 = Maillage_1.GetSubMesh( Ar__te_3, 'bord_3' )
a2D = Maillage_1.GetSubMesh( Assemblage_1, 'Sous-maillage_1' )


## Set names of Mesh objects
smesh.SetName(Regular_1D, 'Regular_1D')
smesh.SetName(Quadrangle_2D, 'Quadrangle_2D')
smesh.SetName(Number_of_Segments_bords_cercles, 'Number of Segments_bords_cercles')
smesh.SetName(Number_of_Segments_arc_cerlces_inf_sup, 'Number of Segments_arc_cerlces_inf_sup')
smesh.SetName(bord_43, 'bord_43')
smesh.SetName(Number_of_Segments_arc_middle, 'Number of Segments_arc_middle')
smesh.SetName(Number_of_Segments_rectanglebas, 'Number of Segments_rectanglebas')
smesh.SetName(bords_rectangles_bas, 'bords_rectangles_bas')
smesh.SetName(bord_2, 'bord_2')
smesh.SetName(bords_horizontaux, 'bords_horizontaux')
smesh.SetName(bord_44, 'bord_44')
smesh.SetName(a2D, '2D')
smesh.SetName(cercles_haut_bas, 'cercles_haut_bas')
smesh.SetName(bords_cercles, 'bords_cercles')
smesh.SetName(cercle_middle, 'cercle_middle')
smesh.SetName(Maillage_1.GetMesh(), 'Maillage_1')
smesh.SetName(bord43, 'bord43')
smesh.SetName(bord2, 'bord2')
smesh.SetName(bord44, 'bord44')
smesh.SetName(bord3, 'bord3')
smesh.SetName(bord_3, 'bord_3')

###
### PARAVIS component
###

import pvsimple
pvsimple.ShowParaviewView()
# trace generated using paraview version 5.6.0-RC1

#### import the simple module from the paraview
from pvsimple import *
#### disable automatic camera reset on 'Show'
pvsimple._DisableFirstRenderCameraReset()

#### saving camera placements for all active views




if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
