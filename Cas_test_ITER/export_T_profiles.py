# trace generated using paraview version 5.7.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`
# to be run with :
# $ pvpython export_T_profiles.py
#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Xdmf3ReaderS'
txdmf = Xdmf3ReaderS(FileName=['/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/T.xdmf'])

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(Input=txdmf,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
point1 = [-1e-9, -0.013500000350177288, 0.0]
point2 = [-1e-9, 0.014499999582767487, 0.0]
plotOverLine1.Source.Point1 = point1
plotOverLine1.Source.Point2 = point2

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1)

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display.ScaleTransferFunction.Points = [373.0212049547773, 0.0, 0.5, 0.0, 1200.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display.OpacityTransferFunction.Points = [373.0212049547773, 0.0, 0.5, 0.0, 1200.0, 1.0, 0.5, 0.0]

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
plotOverLine1Display_1.SeriesVisibility = ['T']
plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', 'T', 'T', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'T', '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', 'T', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesLabelPrefix = ''
plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', 'T', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', 'T', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', 'T', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# get layout
layout1 = GetLayoutByName("Layout #1")

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

# update the view to ensure updated data information
lineChartView1.Update()

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_T_implantation.csv', proxy=plotOverLine1)

animationScene1.GoToNext()

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_T_rest.csv', proxy=plotOverLine1)

animationScene1.GoToNext()

# save data
SaveData('/home/rdelaporte/FESTIM_4_JONATHAN/Cas_test_ITER/results/05_ITER_case_theta_sol2/profile_T_baking.csv', proxy=plotOverLine1)
