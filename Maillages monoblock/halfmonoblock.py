#!/usr/bin/env python

###
### This file is generated automatically by SALOME v9.2.2 with dump python functionality
###

import sys
import salome

salome.salome_init()
theStudy = salome.myStudy
import salome_notebook

notebook = salome_notebook.NoteBook(theStudy)
sys.path.insert(0, r'C:/Users/RD257328/Desktop')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS

h =  0.03/2
l = 0.029
r_coolant = 0.006
r_cucrzr = 0.0075
r_cu = 0.008
geompy = geomBuilder.New(theStudy)

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
Face_1 = geompy.MakeFaceHW(h, l, 1)
Translation_1 = geompy.MakeTranslation(Face_1, -h/2, 0, 0)
Disk_1 = geompy.MakeDiskR(r_cu, 1)
Disk_2 = geompy.MakeDiskR(r_cucrzr, 1)
Disk_3 = geompy.MakeDiskR(r_coolant, 1)
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
geompy.addToStudy( Disk_2, 'Disk_2' )
geompy.addToStudy( Translation_1, 'Translation_1' )
geompy.addToStudy( Disk_1, 'Disk_1' )
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

print('Geometry done.')
###
### SMESH component
###

import  SMESH, SALOMEDS
from salome.smesh import smeshBuilder

ref_heat_flux = 1e-06
ref_coolant = 0.0005


smesh = smeshBuilder.New(theStudy)
Mesh_1 = smesh.Mesh(Partition_1)
NETGEN_1D_2D = Mesh_1.Triangle(algo=smeshBuilder.NETGEN_1D2D)
NETGEN_2D_Parameters_1 = smesh.CreateHypothesis('NETGEN_Parameters_2D', 'libNETGENEngine.so')
NETGEN_2D_Parameters_1.SetMaxSize( 0.00326497 )
NETGEN_2D_Parameters_1.SetSecondOrder( 0 )
NETGEN_2D_Parameters_1.SetOptimize( 1 )
NETGEN_2D_Parameters_1.SetFineness( 2 )
#NETGEN_2D_Parameters_1.SetChordalError( 0.1 )
#NETGEN_2D_Parameters_1.SetChordalErrorEnabled( 0 )
NETGEN_2D_Parameters_1.SetMinSize( 0 )
NETGEN_2D_Parameters_1.SetUseSurfaceCurvature( 1 )
NETGEN_2D_Parameters_1.SetFuseEdges( 1 )
NETGEN_2D_Parameters_1.SetQuadAllowed( 0 )
Regular_1D_2 = Mesh_1.Segment(geom=heat_flux)
LocalLength_1e_06_1e = Regular_1D_2.LocalLength(ref_heat_flux,None,ref_heat_flux/10)
Regular_1D_2_1 = Mesh_1.Segment(geom=coolant)
LocalLength_0_0005_1e = Regular_1D_2_1.LocalLength(ref_coolant,None,ref_heat_flux/10)
isDone = Mesh_1.Compute()

print('Mesh done')

cucrzr_1 = Mesh_1.GroupOnGeom(cucrzr,'cucrzr',SMESH.FACE)
cu_1 = Mesh_1.GroupOnGeom(cu,'cu',SMESH.FACE)
tungsten_1 = Mesh_1.GroupOnGeom(tungsten,'tungsten',SMESH.FACE)
heat_flux_1 = Mesh_1.GroupOnGeom(heat_flux,'heat_flux',SMESH.EDGE)
coolant_1 = Mesh_1.GroupOnGeom(coolant,'coolant',SMESH.EDGE)
left_1 = Mesh_1.GroupOnGeom(left,'left',SMESH.EDGE)
bottom_1 = Mesh_1.GroupOnGeom(bottom,'bottom',SMESH.EDGE)
middle_1 = Mesh_1.GroupOnGeom(middle,'middle',SMESH.EDGE)
smesh.SetName(Mesh_1, 'Mesh_1')
try:
  Mesh_1.ExportMED(r'C:/Users/RD257328/Desktop/halfmonoblock.med',auto_groups=0,minor=33,overwrite=1,meshPart=None,autoDimension=1)
  pass
except:
  print('ExportMED() failed. Invalid file name?')
Regular_1D = Regular_1D_2.GetSubMesh()
Regular_1D_1 = Regular_1D_2_1.GetSubMesh()


## Set names of Mesh objects
smesh.SetName(NETGEN_1D_2D.GetAlgorithm(), 'NETGEN 1D-2D')
smesh.SetName(Regular_1D_2.GetAlgorithm(), 'Regular_1D_2')
smesh.SetName(LocalLength_1e_06_1e, 'LocalLength=1e-06,1e-07')
smesh.SetName(LocalLength_0_0005_1e, 'LocalLength=0.0005,1e-07')
smesh.SetName(NETGEN_2D_Parameters_1, 'NETGEN 2D Parameters_1')
smesh.SetName(cucrzr_1, 'cucrzr')
smesh.SetName(cu_1, 'cu')
smesh.SetName(tungsten_1, 'tungsten')
smesh.SetName(Mesh_1.GetMesh(), 'Mesh_1')
smesh.SetName(Regular_1D_1, 'Regular_1D')
smesh.SetName(heat_flux_1, 'heat_flux')
smesh.SetName(left_1, 'left')
smesh.SetName(coolant_1, 'coolant')
smesh.SetName(middle_1, 'middle')
smesh.SetName(bottom_1, 'bottom')
smesh.SetName(Regular_1D, 'Regular_1D')



if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser()
