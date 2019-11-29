# trace generated using paraview version 5.7.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`
# to be run with :
# $ pvpython export_solute_profiles.py


#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Xdmf3ReaderS'
solute_m3xdmf = Xdmf3ReaderS(FileName=['/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/solute_m3.xdmf'])
solute_m3xdmf.PointArrays = ['solute_m3']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [962, 928]

# show data in view
solute_m3xdmfDisplay = Show(solute_m3xdmf, renderView1)

# get color transfer function/color map for 'solute_m3'
solute_m3LUT = GetColorTransferFunction('solute_m3')

# get opacity transfer function/opacity map for 'solute_m3'
solute_m3PWF = GetOpacityTransferFunction('solute_m3')

# trace defaults for the display properties.
solute_m3xdmfDisplay.Representation = 'Surface'
solute_m3xdmfDisplay.ColorArrayName = ['POINTS', 'solute_m3']
solute_m3xdmfDisplay.LookupTable = solute_m3LUT
solute_m3xdmfDisplay.OSPRayScaleArray = 'solute_m3'
solute_m3xdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
solute_m3xdmfDisplay.SelectOrientationVectors = 'None'
solute_m3xdmfDisplay.ScaleFactor = 0.0027999999932944775
solute_m3xdmfDisplay.SelectScaleArray = 'solute_m3'
solute_m3xdmfDisplay.GlyphType = 'Arrow'
solute_m3xdmfDisplay.GlyphTableIndexArray = 'solute_m3'
solute_m3xdmfDisplay.GaussianRadius = 0.00013999999966472388
solute_m3xdmfDisplay.SetScaleArray = ['POINTS', 'solute_m3']
solute_m3xdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
solute_m3xdmfDisplay.OpacityArray = ['POINTS', 'solute_m3']
solute_m3xdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
solute_m3xdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
solute_m3xdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
solute_m3xdmfDisplay.ScalarOpacityFunction = solute_m3PWF
solute_m3xdmfDisplay.ScalarOpacityUnitDistance = 0.0009165247490795945

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
solute_m3xdmfDisplay.ScaleTransferFunction.Points = [-5.665308217800296e+21, 0.0, 0.5, 0.0, 2.7121756018495845e+23, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
solute_m3xdmfDisplay.OpacityTransferFunction.Points = [-5.665308217800296e+21, 0.0, 0.5, 0.0, 2.7121756018495845e+23, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
solute_m3xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(Input=solute_m3xdmf,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine1.Source.Point1 = [-0.014000000432133675, -0.013500000350177288, 0.0]
plotOverLine1.Source.Point2 = [1.6488543198088203e-16, 0.014499999582767487, 0.0]

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1)

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'
plotOverLine1Display.ColorArrayName = ['POINTS', 'solute_m3']
plotOverLine1Display.LookupTable = solute_m3LUT
plotOverLine1Display.OSPRayScaleArray = 'solute_m3'
plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine1Display.SelectOrientationVectors = 'None'
plotOverLine1Display.ScaleFactor = 0.0027999999932944775
plotOverLine1Display.SelectScaleArray = 'solute_m3'
plotOverLine1Display.GlyphType = 'Arrow'
plotOverLine1Display.GlyphTableIndexArray = 'solute_m3'
plotOverLine1Display.GaussianRadius = 0.00013999999966472388
plotOverLine1Display.SetScaleArray = ['POINTS', 'solute_m3']
plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.OpacityArray = ['POINTS', 'solute_m3']
plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display.ScaleTransferFunction.Points = [-112803296.0, 0.0, 0.5, 0.0, 2.7028628783961227e+23, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display.OpacityTransferFunction.Points = [-112803296.0, 0.0, 0.5, 0.0, 2.7028628783961227e+23, 1.0, 0.5, 0.0]

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
# uncomment following to set a specific view size
# lineChartView1.ViewSize = [400, 400]

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1)

# trace defaults for the display properties.
plotOverLine1Display_1.CompositeDataSetIndex = [0]
plotOverLine1Display_1.UseIndexForXAxis = 0
plotOverLine1Display_1.XArrayName = 'arc_length'
plotOverLine1Display_1.SeriesVisibility = ['solute_m3']
plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', 'solute_m3', 'solute_m3', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'solute_m3', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', 'solute_m3', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesLabelPrefix = ''
plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', 'solute_m3', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', 'solute_m3', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', 'solute_m3', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# update the view to ensure updated data information
lineChartView1.Update()

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_implantation.csv', proxy=plotOverLine1)

animationScene1.GoToNext()

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_rest.csv', proxy=plotOverLine1)

animationScene1.GoToNext()

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_baking.csv', proxy=plotOverLine1)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-0.007000000216066755, 0.0004999996162950993, 0.06047652290401769]
renderView1.CameraFocalPoint = [-0.007000000216066755, 0.0004999996162950993, 0.0]
renderView1.CameraParallelScale = 0.015652475909138583

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).