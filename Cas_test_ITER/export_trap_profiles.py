# trace generated using paraview version 5.7.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Xdmf3ReaderS'
a1xdmf = Xdmf3ReaderS(FileName=['/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/1.xdmf'])

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# create a new 'Xdmf3ReaderS'
a2xdmf = Xdmf3ReaderS(FileName=['/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/2.xdmf'])

# create a new 'Xdmf3ReaderS'
a3xdmf = Xdmf3ReaderS(FileName=['/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/3.xdmf'])

# create a new 'Xdmf3ReaderS'
a4xdmf = Xdmf3ReaderS(FileName=['/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/4.xdmf'])

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraFocalDisk = 1.0
renderView1.Background = [0.32, 0.34, 0.43]
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1
# uncomment following to set a specific view size
# renderView1.ViewSize = [400, 400]

# show data in view
a1xdmfDisplay = Show(a1xdmf, renderView1)

# get color transfer function/color map for 'a1'
a1LUT = GetColorTransferFunction('a1')

# get opacity transfer function/opacity map for 'a1'
a1PWF = GetOpacityTransferFunction('a1')

# trace defaults for the display properties.
a1xdmfDisplay.Representation = 'Surface'
a1xdmfDisplay.ColorArrayName = ['POINTS', '1']
a1xdmfDisplay.LookupTable = a1LUT
a1xdmfDisplay.OSPRayScaleArray = '1'
a1xdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
a1xdmfDisplay.SelectOrientationVectors = '1'
a1xdmfDisplay.ScaleFactor = 0.0027999999932944775
a1xdmfDisplay.SelectScaleArray = '1'
a1xdmfDisplay.GlyphType = 'Arrow'
a1xdmfDisplay.GlyphTableIndexArray = '1'
a1xdmfDisplay.GaussianRadius = 0.00013999999966472388
a1xdmfDisplay.SetScaleArray = ['POINTS', '1']
a1xdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
a1xdmfDisplay.OpacityArray = ['POINTS', '1']
a1xdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
a1xdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
a1xdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
a1xdmfDisplay.ScalarOpacityFunction = a1PWF
a1xdmfDisplay.ScalarOpacityUnitDistance = 0.0009165247490795945

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a1xdmfDisplay.ScaleTransferFunction.Points = [-2.160247397868329e+23, 0.0, 0.5, 0.0, 4.725764430774098e+24, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a1xdmfDisplay.OpacityTransferFunction.Points = [-2.160247397868329e+23, 0.0, 0.5, 0.0, 4.725764430774098e+24, 1.0, 0.5, 0.0]

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=renderView1, layout=layout1, hint=0)

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [-0.007000000216066755, 0.0004999996162950993, 10000.0]
renderView1.CameraFocalPoint = [-0.007000000216066755, 0.0004999996162950993, 0.0]

# show color bar/color legend
a1xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
a4xdmfDisplay = Show(a4xdmf, renderView1)

# get color transfer function/color map for 'a4'
a4LUT = GetColorTransferFunction('a4')

# get opacity transfer function/opacity map for 'a4'
a4PWF = GetOpacityTransferFunction('a4')

# trace defaults for the display properties.
a4xdmfDisplay.Representation = 'Surface'
a4xdmfDisplay.ColorArrayName = ['POINTS', '4']
a4xdmfDisplay.LookupTable = a4LUT
a4xdmfDisplay.OSPRayScaleArray = '4'
a4xdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
a4xdmfDisplay.SelectOrientationVectors = '4'
a4xdmfDisplay.ScaleFactor = 0.0027999999932944775
a4xdmfDisplay.SelectScaleArray = '4'
a4xdmfDisplay.GlyphType = 'Arrow'
a4xdmfDisplay.GlyphTableIndexArray = '4'
a4xdmfDisplay.GaussianRadius = 0.00013999999966472388
a4xdmfDisplay.SetScaleArray = ['POINTS', '4']
a4xdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
a4xdmfDisplay.OpacityArray = ['POINTS', '4']
a4xdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
a4xdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
a4xdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
a4xdmfDisplay.ScalarOpacityFunction = a4PWF
a4xdmfDisplay.ScalarOpacityUnitDistance = 0.0009165247490795945

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a4xdmfDisplay.ScaleTransferFunction.Points = [-4.310319428582534e-15, 0.0, 0.5, 0.0, 1.6409844775243355e+21, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a4xdmfDisplay.OpacityTransferFunction.Points = [-4.310319428582534e-15, 0.0, 0.5, 0.0, 1.6409844775243355e+21, 1.0, 0.5, 0.0]

# show color bar/color legend
a4xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
a3xdmfDisplay = Show(a3xdmf, renderView1)

# get color transfer function/color map for 'a3'
a3LUT = GetColorTransferFunction('a3')

# get opacity transfer function/opacity map for 'a3'
a3PWF = GetOpacityTransferFunction('a3')

# trace defaults for the display properties.
a3xdmfDisplay.Representation = 'Surface'
a3xdmfDisplay.ColorArrayName = ['POINTS', '3']
a3xdmfDisplay.LookupTable = a3LUT
a3xdmfDisplay.OSPRayScaleArray = '3'
a3xdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
a3xdmfDisplay.SelectOrientationVectors = '3'
a3xdmfDisplay.ScaleFactor = 0.0027999999932944775
a3xdmfDisplay.SelectScaleArray = '3'
a3xdmfDisplay.GlyphType = 'Arrow'
a3xdmfDisplay.GlyphTableIndexArray = '3'
a3xdmfDisplay.GaussianRadius = 0.00013999999966472388
a3xdmfDisplay.SetScaleArray = ['POINTS', '3']
a3xdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
a3xdmfDisplay.OpacityArray = ['POINTS', '3']
a3xdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
a3xdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
a3xdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
a3xdmfDisplay.ScalarOpacityFunction = a3PWF
a3xdmfDisplay.ScalarOpacityUnitDistance = 0.0009165247490795945

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a3xdmfDisplay.ScaleTransferFunction.Points = [-0.062445566058158875, 0.0, 0.5, 0.0, 1.1701103018252645e+20, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a3xdmfDisplay.OpacityTransferFunction.Points = [-0.062445566058158875, 0.0, 0.5, 0.0, 1.1701103018252645e+20, 1.0, 0.5, 0.0]

# show color bar/color legend
a3xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# show data in view
a2xdmfDisplay = Show(a2xdmf, renderView1)

# get color transfer function/color map for 'a2'
a2LUT = GetColorTransferFunction('a2')

# get opacity transfer function/opacity map for 'a2'
a2PWF = GetOpacityTransferFunction('a2')

# trace defaults for the display properties.
a2xdmfDisplay.Representation = 'Surface'
a2xdmfDisplay.ColorArrayName = ['POINTS', '2']
a2xdmfDisplay.LookupTable = a2LUT
a2xdmfDisplay.OSPRayScaleArray = '2'
a2xdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
a2xdmfDisplay.SelectOrientationVectors = '2'
a2xdmfDisplay.ScaleFactor = 0.0027999999932944775
a2xdmfDisplay.SelectScaleArray = '2'
a2xdmfDisplay.GlyphType = 'Arrow'
a2xdmfDisplay.GlyphTableIndexArray = '2'
a2xdmfDisplay.GaussianRadius = 0.00013999999966472388
a2xdmfDisplay.SetScaleArray = ['POINTS', '2']
a2xdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
a2xdmfDisplay.OpacityArray = ['POINTS', '2']
a2xdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
a2xdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
a2xdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
a2xdmfDisplay.ScalarOpacityFunction = a2PWF
a2xdmfDisplay.ScalarOpacityUnitDistance = 0.0009165247490795945

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
a2xdmfDisplay.ScaleTransferFunction.Points = [-5.9202825506336253e+23, 0.0, 0.5, 0.0, 1.552884674304146e+24, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
a2xdmfDisplay.OpacityTransferFunction.Points = [-5.9202825506336253e+23, 0.0, 0.5, 0.0, 1.552884674304146e+24, 1.0, 0.5, 0.0]

# show color bar/color legend
a2xdmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(a1xdmf)

# destroy renderView1
Delete(renderView1)
del renderView1

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(Input=a1xdmf,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine1.Source.Point1 = [-0.014000000432133675, -0.013500000350177288, 0.0]
plotOverLine1.Source.Point2 = [1.6488543198088203e-16, 0.014499999582767487, 0.0]

# set active source
SetActiveSource(a2xdmf)

# create a new 'Plot Over Line'
plotOverLine2 = PlotOverLine(Input=a2xdmf,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine2.Source.Point1 = [-0.014000000432133675, -0.013500000350177288, 0.0]
plotOverLine2.Source.Point2 = [1.6488543198088203e-16, 0.014499999582767487, 0.0]

# set active source
SetActiveSource(a3xdmf)

# create a new 'Plot Over Line'
plotOverLine3 = PlotOverLine(Input=a3xdmf,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine3.Source.Point1 = [-0.014000000432133675, -0.013500000350177288, 0.0]
plotOverLine3.Source.Point2 = [1.6488543198088203e-16, 0.014499999582767487, 0.0]

# set active source
SetActiveSource(a4xdmf)

# create a new 'Plot Over Line'
plotOverLine4 = PlotOverLine(Input=a4xdmf,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine4.Source.Point1 = [-0.014000000432133675, -0.013500000350177288, 0.0]
plotOverLine4.Source.Point2 = [1.6488543198088203e-16, 0.014499999582767487, 0.0]

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
# uncomment following to set a specific view size
# lineChartView1.ViewSize = [400, 400]

# show data in view
plotOverLine3Display = Show(plotOverLine3, lineChartView1)

# trace defaults for the display properties.
plotOverLine3Display.CompositeDataSetIndex = [0]
plotOverLine3Display.UseIndexForXAxis = 0
plotOverLine3Display.XArrayName = 'arc_length'
plotOverLine3Display.SeriesVisibility = ['3']
plotOverLine3Display.SeriesLabel = ['3', '3', 'arc_length', 'arc_length', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine3Display.SeriesColor = ['3', '0', '0', '0', 'arc_length', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
plotOverLine3Display.SeriesPlotCorner = ['3', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine3Display.SeriesLabelPrefix = ''
plotOverLine3Display.SeriesLineStyle = ['3', '1', 'arc_length', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine3Display.SeriesLineThickness = ['3', '2', 'arc_length', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine3Display.SeriesMarkerStyle = ['3', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# show data in view
plotOverLine4Display = Show(plotOverLine4, lineChartView1)

# trace defaults for the display properties.
plotOverLine4Display.CompositeDataSetIndex = [0]
plotOverLine4Display.UseIndexForXAxis = 0
plotOverLine4Display.XArrayName = 'arc_length'
plotOverLine4Display.SeriesVisibility = ['4']
plotOverLine4Display.SeriesLabel = ['4', '4', 'arc_length', 'arc_length', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine4Display.SeriesColor = ['4', '0', '0', '0', 'arc_length', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
plotOverLine4Display.SeriesPlotCorner = ['4', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine4Display.SeriesLabelPrefix = ''
plotOverLine4Display.SeriesLineStyle = ['4', '1', 'arc_length', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine4Display.SeriesLineThickness = ['4', '2', 'arc_length', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine4Display.SeriesMarkerStyle = ['4', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# show data in view
plotOverLine2Display = Show(plotOverLine2, lineChartView1)

# trace defaults for the display properties.
plotOverLine2Display.CompositeDataSetIndex = [0]
plotOverLine2Display.UseIndexForXAxis = 0
plotOverLine2Display.XArrayName = 'arc_length'
plotOverLine2Display.SeriesVisibility = ['2']
plotOverLine2Display.SeriesLabel = ['2', '2', 'arc_length', 'arc_length', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine2Display.SeriesColor = ['2', '0', '0', '0', 'arc_length', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
plotOverLine2Display.SeriesPlotCorner = ['2', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine2Display.SeriesLabelPrefix = ''
plotOverLine2Display.SeriesLineStyle = ['2', '1', 'arc_length', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine2Display.SeriesLineThickness = ['2', '2', 'arc_length', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine2Display.SeriesMarkerStyle = ['2', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# show data in view
plotOverLine1Display = Show(plotOverLine1, lineChartView1)

# trace defaults for the display properties.
plotOverLine1Display.CompositeDataSetIndex = [0]
plotOverLine1Display.UseIndexForXAxis = 0
plotOverLine1Display.XArrayName = 'arc_length'
plotOverLine1Display.SeriesVisibility = ['1']
plotOverLine1Display.SeriesLabel = ['1', '1', 'arc_length', 'arc_length', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display.SeriesColor = ['1', '0', '0', '0', 'arc_length', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
plotOverLine1Display.SeriesPlotCorner = ['1', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display.SeriesLabelPrefix = ''
plotOverLine1Display.SeriesLineStyle = ['1', '1', 'arc_length', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display.SeriesLineThickness = ['1', '2', 'arc_length', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display.SeriesMarkerStyle = ['1', '0', 'arc_length', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# update the view to ensure updated data information
lineChartView1.Update()

# set active source
SetActiveSource(plotOverLine3)

# set active source
SetActiveSource(plotOverLine2)

# set active source
SetActiveSource(plotOverLine1)

# update the view to ensure updated data information
lineChartView1.Update()

# set active source
SetActiveSource(plotOverLine1)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap1_implantation.csv', proxy=plotOverLine1)

# set active source
SetActiveSource(plotOverLine2)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap2_implantation.csv', proxy=plotOverLine2)

# set active source
SetActiveSource(plotOverLine3)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap3_implantation.csv', proxy=plotOverLine3)

# set active source
SetActiveSource(plotOverLine4)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap4_implantation.csv', proxy=plotOverLine4)

animationScene1.GoToNext()

# set active source
SetActiveSource(plotOverLine1)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap1_rest.csv', proxy=plotOverLine1)

# set active source
SetActiveSource(plotOverLine2)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap2_rest.csv', proxy=plotOverLine2)

# set active source
SetActiveSource(plotOverLine3)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap3_rest.csv', proxy=plotOverLine3)

# set active source
SetActiveSource(plotOverLine4)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap4_rest.csv', proxy=plotOverLine4)

animationScene1.GoToNext()

# set active source
SetActiveSource(plotOverLine1)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap1_baking.csv', proxy=plotOverLine1)

# set active source
SetActiveSource(plotOverLine2)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap2_baking.csv', proxy=plotOverLine2)

# set active source
SetActiveSource(plotOverLine3)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap3_baking.csv', proxy=plotOverLine3)

# set active source
SetActiveSource(plotOverLine4)

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_trap4_baking.csv', proxy=plotOverLine4)

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).