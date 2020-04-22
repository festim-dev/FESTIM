# trace generated using paraview version 5.7.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#import sys

#print(sys.path)

#sys.path.insert(0, 'C:\\Program Files\\ParaView 5.7.0-Windows-Python3.7-msvc2015-64bit\\bin\\Lib\\site-packages')
#sys.path.insert(0, 'C:\\Program Files\\ParaView 5.7.0-Windows-Python3.7-msvc2015-64bit\\bin\\Lib')
#sys.path.insert(0, 'C:\\Program Files\\ParaView 5.7.0-Windows-Python3.7-msvc2015-64bit\\bin')

#print(sys.path)

varnames = ['1','2','3','4','solute_m3','T','retention_m3','theta']

directory= "C:\\Users\\jmougenot\\Desktop\\FESTIM_4_JONATHAN\\Cas_test_ITER\\results\\ITER_case_theta_sol2_99950\\"
    
#varname='1'
#varname='2'
#varname='3'
#varname='4'
#varname='solute_m3'
#varname='T'
#varname='retention_m3'

#varaxe=1  #y droit
#varaxe=2 # y milieu

for varaxe in [1,2]:

    for varname in varnames:
            

        #### import the simple module from the paraview
        from paraview.simple import *
        #### disable automatic camera reset on 'Show'
        paraview.simple._DisableFirstRenderCameraReset()

        # create a new 'Xdmf3ReaderS'
        txdmf = Xdmf3ReaderS(FileName=[directory+varname+'.xdmf'])

        # get animation scene
        animationScene1 = GetAnimationScene()

        # get the time-keeper
        timeKeeper1 = GetTimeKeeper()

        # update animation scene based on data timesteps
        animationScene1.UpdateAnimationUsingDataTimeSteps()

        # get active view
        renderView1 = GetActiveViewOrCreate('RenderView')
        # uncomment following to set a specific view size
        # renderView1.ViewSize = [1082, 782]

        # show data in view
        txdmfDisplay = Show(txdmf, renderView1)


        # get color transfer function/color map for 'T'
        tLUT = GetColorTransferFunction(varname)

        # get opacity transfer function/opacity map for 'T'
        tPWF = GetOpacityTransferFunction(varname)

        # trace defaults for the display properties.
        txdmfDisplay.Representation = 'Surface'
        txdmfDisplay.ColorArrayName = ['POINTS', varname]
        txdmfDisplay.LookupTable = tLUT
        txdmfDisplay.OSPRayScaleArray = varname
        txdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
        txdmfDisplay.SelectOrientationVectors = 'None'
        txdmfDisplay.ScaleFactor = 0.0027999999932944775
        txdmfDisplay.SelectScaleArray = varname
        txdmfDisplay.GlyphType = 'Arrow'
        txdmfDisplay.GlyphTableIndexArray = varname
        txdmfDisplay.GaussianRadius = 0.00013999999966472388
        txdmfDisplay.SetScaleArray = ['POINTS', varname]
        txdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
        txdmfDisplay.OpacityArray = ['POINTS', varname]
        txdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
        txdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
        txdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
        txdmfDisplay.ScalarOpacityFunction = tPWF
        txdmfDisplay.ScalarOpacityUnitDistance = 0.000674557186611405

        # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
        txdmfDisplay.ScaleTransferFunction.Points = [373.0, 0.0, 0.5, 0.0, 1200.0, 1.0, 0.5, 0.0]

        # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
        txdmfDisplay.OpacityTransferFunction.Points = [373.0, 0.0, 0.5, 0.0, 1200.0, 1.0, 0.5, 0.0]

        # reset view to fit data
        renderView1.ResetCamera()

        #changing interaction mode based on data extents
        renderView1.InteractionMode = '2D'
        renderView1.CameraPosition = [-0.007000000216066755, 0.0004999996162950993, 10000.0]
        renderView1.CameraFocalPoint = [-0.007000000216066755, 0.0004999996162950993, 0.0]

        # get the material library
        materialLibrary1 = GetMaterialLibrary()

        # show color bar/color legend
        txdmfDisplay.SetScalarBarVisibility(renderView1, True)

        # update the view to ensure updated data information
        renderView1.Update()

        # create a new 'Plot Over Line'
        plotOverLine1 = PlotOverLine(Input=txdmf,
            Source='High Resolution Line Source')

        # init the 'High Resolution Line Source' selected for 'Source'
        if varaxe==1:
            plotOverLine1.Source.Point1 = [-1.0e-6, -0.0135, 0.0]
            plotOverLine1.Source.Point2 = [-1.0e-6, 0.0145, 0.0]
        elif varaxe==2:
            plotOverLine1.Source.Point1 = [-0.007, -0.0135, 0.0]
            plotOverLine1.Source.Point2 = [-0.007, 0.0145, 0.0]
            

        # show data in view
        plotOverLine1Display = Show(plotOverLine1, renderView1)

        # trace defaults for the display properties.
        plotOverLine1Display.Representation = 'Surface'
        plotOverLine1Display.ColorArrayName = ['POINTS', varname]
        plotOverLine1Display.LookupTable = tLUT
        plotOverLine1Display.OSPRayScaleArray = varname
        plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
        plotOverLine1Display.SelectOrientationVectors = 'None'
        plotOverLine1Display.ScaleFactor = 0.0027999999932944775
        plotOverLine1Display.SelectScaleArray = varname
        plotOverLine1Display.GlyphType = 'Arrow'
        plotOverLine1Display.GlyphTableIndexArray = varname
        plotOverLine1Display.GaussianRadius = 0.00013999999966472388
        plotOverLine1Display.SetScaleArray = ['POINTS', varname]
        plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
        plotOverLine1Display.OpacityArray = ['POINTS', varname]
        plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
        plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
        plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'

        # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
        plotOverLine1Display.ScaleTransferFunction.Points = [373.02140395638133, 0.0, 0.5, 0.0, 1200.0, 1.0, 0.5, 0.0]

        # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
        plotOverLine1Display.OpacityTransferFunction.Points = [373.02140395638133, 0.0, 0.5, 0.0, 1200.0, 1.0, 0.5, 0.0]

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
        plotOverLine1Display_1.SeriesVisibility = [varname]
        plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', varname, varname, 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
        plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', varname, '0.89', '0.1', '0.11', 'vtkValidPointMask', '0.22', '0.49', '0.72', 'Points_X', '0.3', '0.69', '0.29', 'Points_Y', '0.6', '0.31', '0.64', 'Points_Z', '1', '0.5', '0', 'Points_Magnitude', '0.65', '0.34', '0.16']
        plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', varname, '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
        plotOverLine1Display_1.SeriesLabelPrefix = ''
        plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', varname, '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
        plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', varname, '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
        plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', varname, '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

        # get layout
        layout1 = GetLayoutByName("Layout #1")

        # add view to a layout so it's visible in UI
        AssignViewToLayout(view=lineChartView1, layout=layout1, hint=0)

        # Properties modified on plotOverLine1Display_1
        plotOverLine1Display_1.XArrayName = 'Points_Y'

        # export view
        ExportView(directory+'/extract001/'+varname+'_'+str(varaxe)+'_01.csv', view=lineChartView1)

        # Properties modified on animationScene1
        #animationScene1.AnimationTime = 2400000.0

        # Properties modified on timeKeeper1
        #timeKeeper1.Time = 2400000.0
        animationScene1.GoToNext()

        # export view
        ExportView(directory+'/extract001/'+varname+'_'+str(varaxe)+'_02.csv', view=lineChartView1)

        # Properties modified on animationScene1
        #animationScene1.AnimationTime = 7800000.0

        # Properties modified on timeKeeper1
        #timeKeeper1.Time = 7800000.0
        animationScene1.GoToNext()

        # export view
        ExportView(directory+'/extract001/'+varname+'_'+str(varaxe)+'_03.csv', view=lineChartView1)

        # Properties modified on animationScene1
        #animationScene1.AnimationTime = 13200000.0

        # Properties modified on timeKeeper1
        #timeKeeper1.Time = 13200000.0
        animationScene1.GoToNext()

        # export view
        ExportView(directory+'/extract001/'+varname+'_'+str(varaxe)+'_04.csv', view=lineChartView1)

        # Properties modified on animationScene1
        #animationScene1.AnimationTime = 14496000.0

        # Properties modified on timeKeeper1
        #timeKeeper1.Time = 14496000.0
        animationScene1.GoToNext()

        # export view
        ExportView(directory+'/extract001/'+varname+'_'+str(varaxe)+'_05.csv', view=lineChartView1)

        # Properties modified on animationScene1
        #animationScene1.AnimationTime = 15792000.0

        # Properties modified on timeKeeper1
        #timeKeeper1.Time = 15792000.0
        animationScene1.GoToNext()

        # export view
        ExportView(directory+'/extract001/'+varname+'_'+str(varaxe)+'_06.csv', view=lineChartView1)

        #### saving camera placements for all active views

        # current camera placement for renderView1
        renderView1.InteractionMode = '2D'
        renderView1.CameraPosition = [-0.007000000216066755, 0.0004999996162950993, 10000.0]
        renderView1.CameraFocalPoint = [-0.007000000216066755, 0.0004999996162950993, 0.0]
        renderView1.CameraParallelScale = 0.015652475909138583

        #### uncomment the following to render all views
        # RenderAllViews()
        # alternatively, if you want to write images, you can use SaveScreenshot(...).