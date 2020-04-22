#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.3.0 with dump python functionality
###

import sys
import salome

salome.salome_init()
import salome_notebook
notebook = salome_notebook.NoteBook()
sys.path.insert(0, r'D:/Logiciels/FESTIM_4_JONATHAN/Cas_test_ITER/maillages/Mesh_ITER_58734')

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
Face_1 = geompy.MakeFaceHW(0.014, 0.028, 1)
Translation_1 = geompy.MakeTranslation(Face_1, -0.007, 0.0005, 0)
Disk_1 = geompy.MakeDiskR(0.008500000000000001, 1)
Disk_2 = geompy.MakeDiskR(0.0075, 1)
Disk_3 = geompy.MakeDiskR(0.006, 1)
Common_1 = geompy.MakeCommonList([Translation_1, Disk_1], True)
Cut_1 = geompy.MakeCutList(Translation_1, [Common_1], True)
Common_2 = geompy.MakeCommonList([Disk_2, Common_1], True)
Cut_2 = geompy.MakeCutList(Common_1, [Common_2], True)
Common_3 = geompy.MakeCommonList([Translation_1, Disk_3], True)
Cut_3 = geompy.MakeCutList(Common_2, [Common_3], True)
Partition_1 = geompy.MakePartition([Cut_1, Cut_2, Cut_3], [], [], [], geompy.ShapeType["FACE"], 0, [], 0)
cucrzr = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
geompy.UnionIDs(cucrzr, [27])
cu = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
geompy.UnionIDs(cu, [18])
tungsten = geompy.CreateGroup(Partition_1, geompy.ShapeType["FACE"])
geompy.UnionIDs(tungsten, [2])
heat_flux = geompy.CreateGroup(Partition_1, geompy.ShapeType["EDGE"])
geompy.UnionIDs(heat_flux, [7])
coolant = geompy.CreateGroup(Partition_1, geompy.ShapeType["EDGE"])
geompy.UnionIDs(coolant, [33, 31])
left = geompy.CreateGroup(Partition_1, geompy.ShapeType["EDGE"])
geompy.UnionIDs(left, [4])
bottom = geompy.CreateGroup(Partition_1, geompy.ShapeType["EDGE"])
geompy.UnionIDs(bottom, [17])
middle = geompy.CreateGroup(Partition_1, geompy.ShapeType["EDGE"])
geompy.UnionIDs(middle, [15, 9, 26, 35, 29, 20])
geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )
geompy.addToStudy( Face_1, 'Face_1' )
geompy.addToStudy( Translation_1, 'Translation_1' )
geompy.addToStudy( Disk_1, 'Disk_1' )
geompy.addToStudy( Disk_2, 'Disk_2' )
geompy.addToStudy( Disk_3, 'Disk_3' )
geompy.addToStudy( Common_1, 'Common_1' )
geompy.addToStudy( Cut_1, 'Cut_1' )
geompy.addToStudy( Common_2, 'Common_2' )
geompy.addToStudy( Cut_2, 'Cut_2' )
geompy.addToStudy( Common_3, 'Common_3' )
geompy.addToStudy( Cut_3, 'Cut_3' )
geompy.addToStudy( Partition_1, 'Partition_1' )
geompy.addToStudyInFather( Partition_1, cucrzr, 'cucrzr' )
geompy.addToStudyInFather( Partition_1, cu, 'cu' )
geompy.addToStudyInFather( Partition_1, tungsten, 'tungsten' )
geompy.addToStudyInFather( Partition_1, heat_flux, 'heat_flux' )
geompy.addToStudyInFather( Partition_1, coolant, 'coolant' )
geompy.addToStudyInFather( Partition_1, left, 'left' )
geompy.addToStudyInFather( Partition_1, bottom, 'bottom' )
geompy.addToStudyInFather( Partition_1, middle, 'middle' )

###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

smesh = smeshBuilder.New()
#smesh.SetEnablePublish( False ) # Set to False to avoid publish in study if not needed or in some particular situations:
                                 # multiples meshes built in parallel, complex and numerous mesh edition (performance)

Mesh_1 = smesh.Mesh(Partition_1)
NETGEN_1D_2D = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_1D2D)
Regular_1D_2 = Mesh_1.Segment(geom=heat_flux)
Regular_1D_2_1 = Mesh_1.Segment(geom=coolant)
cucrzr_1 = Mesh_1.GroupOnGeom(cucrzr,'cucrzr',SMESH.FACE)
cu_1 = Mesh_1.GroupOnGeom(cu,'cu',SMESH.FACE)
tungsten_1 = Mesh_1.GroupOnGeom(tungsten,'tungsten',SMESH.FACE)
heat_flux_1 = Mesh_1.GroupOnGeom(heat_flux,'heat_flux',SMESH.EDGE)
coolant_1 = Mesh_1.GroupOnGeom(coolant,'coolant',SMESH.EDGE)
left_1 = Mesh_1.GroupOnGeom(left,'left',SMESH.EDGE)
bottom_1 = Mesh_1.GroupOnGeom(bottom,'bottom',SMESH.EDGE)
middle_1 = Mesh_1.GroupOnGeom(middle,'middle',SMESH.EDGE)
NETGEN_2D_Parameters_1 = NETGEN_1D_2D.Parameters()
NETGEN_2D_Parameters_1.SetUseDelauney( 0 )
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cucrzr, 0.000326497)
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cu, 0.000326497)
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(heat_flux, 1e-06)
NETGEN_2D_Parameters_1.UnsetLocalSizeOnEntry("0:1:1:16:9")
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(heat_flux, 5e-06)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
Regular_1D_2_2 = Mesh_1.Segment(geom=left)
Start_and_End_Length_1 = Regular_1D_2_2.StartEndLength(0.0021838,9e-08,[])
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
cu_1.SetColor( SALOMEDS.Color( 1, 0.666667, 0 ))
tungsten_1.SetColor( SALOMEDS.Color( 1, 0.666667, 0 ))
cucrzr_1.SetColor( SALOMEDS.Color( 1, 0.666667, 0 ))
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cucrzr, 5.26497e-05)
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cu, 5.26497e-05)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cucrzr, 0.000526497)
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cu, 0.000526497)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cucrzr, 0.000226497)
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cu, 0.000226497)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.UnsetLocalSizeOnEntry("0:1:1:16:7")
Regular_1D_2_3 = Mesh_1.Segment(geom=heat_flux)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
Start_and_End_Length_2 = Regular_1D_2.StartEndLength(9e-08,5e-05,[])
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
Start_and_End_Length_2.SetObjectEntry( "0:1:1:16" )
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
smesh.SetName(Mesh_1, 'Mesh_1')
try:
  Mesh_1.ExportMED(r'D:/Logiciels/FESTIM_4_JONATHAN/Cas_test_ITER/maillages/Mesh_ITER/Mesh_ITER_50798.med',auto_groups=0,minor=40,overwrite=1,meshPart=None,autoDimension=1)
  pass
except:
  print('ExportMED() failed. Invalid file name?')
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetMinSize( 0 )
NETGEN_2D_Parameters_1.SetSecondOrder( 0 )
NETGEN_2D_Parameters_1.SetOptimize( 1 )
NETGEN_2D_Parameters_1.SetChordalError( 0 )
NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_2D_Parameters_1.SetFuseEdges( 1 )
NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
NETGEN_2D_Parameters_1.SetLocalSizeOnShape(cucrzr, 0.000226497)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
status = Mesh_1.RemoveHypothesis(Start_and_End_Length_1,left)
status = Mesh_1.AddHypothesis(Start_and_End_Length_1,left)
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetNbSegPerEdge( 3 )
NETGEN_2D_Parameters_1.SetNbSegPerRadius( 5 )
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetMaxSize( 0.00019 )
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
Start_and_End_Length_1.SetStartLength( 0.0003 )
Start_and_End_Length_1.SetEndLength( 9e-08 )
Start_and_End_Length_1.SetReversedEdges( [] )
Start_and_End_Length_1.SetObjectEntry( "0:1:1:16" )
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
NETGEN_2D_Parameters_1.SetFineness( 5 )
NETGEN_2D_Parameters_1.SetGrowthRate( 0.15 )
NETGEN_2D_Parameters_1.SetWorstElemMeasure( 0 )
NETGEN_2D_Parameters_1.SetCheckChartBoundary( 200 )
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
isDone = Mesh_1.Compute()
[ cucrzr_1, cu_1, tungsten_1, heat_flux_1, coolant_1, left_1, bottom_1, middle_1 ] = Mesh_1.GetGroups()
smesh.SetName(Mesh_1, 'Mesh_1')
try:
  Mesh_1.ExportMED(r'D:/Logiciels/FESTIM_4_JONATHAN/Cas_test_ITER/maillages/Mesh_ITER_58734/Mesh_ITER.med',auto_groups=0,minor=40,overwrite=1,meshPart=None,autoDimension=1)
  pass
except:
  print('ExportMED() failed. Invalid file name?')
Regular_1D = Regular_1D_2_1.GetSubMesh()
left_2 = Regular_1D_2_2.GetSubMesh()
top = Regular_1D_2.GetSubMesh()


## Set names of Mesh objects
smesh.SetName(NETGEN_1D_2D.GetAlgorithm(), 'NETGEN 1D-2D')
smesh.SetName(Regular_1D_2.GetAlgorithm(), 'Regular_1D_2')
smesh.SetName(Start_and_End_Length_1, 'Start and End Length_1')
smesh.SetName(Start_and_End_Length_2, 'Start and End Length_2')
smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
smesh.SetName(cucrzr_1, 'cucrzr')
smesh.SetName(cu_1, 'cu')
smesh.SetName(tungsten_1, 'tungsten')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(Regular_1D, 'Regular_1D')
smesh.SetName(heat_flux_1, 'heat_flux')
smesh.SetName(left_1, 'left')
smesh.SetName(coolant_1, 'coolant')
smesh.SetName(middle_1, 'middle')
smesh.SetName(bottom_1, 'bottom')
smesh.SetName(left_2, 'left')
smesh.SetName(top, 'top')


if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
