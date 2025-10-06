import os

def create_cavity_case(case_path):
    """Creates a self-contained OpenFOAM cavity case directory."""
    os.makedirs(os.path.join(case_path, 'system'), exist_ok=True)
    os.makedirs(os.path.join(case_path, 'constant'), exist_ok=True)
    os.makedirs(os.path.join(case_path, '0'), exist_ok=True)

    # --- File Contents ---
    blockMeshDict_content = """
    FoamFile { format ascii; class dictionary; object blockMeshDict; }
    convertToMeters 1;
    vertices ( (0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1) );
    blocks ( hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1) );
    edges ();
    boundary (
        movingWall { type wall; faces ((3 7 6 2)); }
        fixedWalls { type wall; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }
        frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
    );
    mergePatchPairs ();
    """

    controlDict_content = """
    FoamFile { format ascii; class dictionary; location "system"; object controlDict; }
    application     simpleFoam;
    startFrom       latestTime;
    startTime       0;
    stopAt          endTime;
    endTime         1000;
    deltaT          1;
    writeControl    timeStep;
    writeInterval   50;
    purgeWrite      0;
    timeFormat      general;
    runTimeModifiable true;
    """

    fvSchemes_content = """
    FoamFile { format ascii; class dictionary; location "system"; object fvSchemes; }
    ddtSchemes { default steadyState; }
    gradSchemes { default Gauss linear; }
    divSchemes {
        default         none;
        div(phi,U)      Gauss upwind;
        div((nuEff*dev2(T(grad(U))))) Gauss linear;
    }
    laplacianSchemes { default Gauss linear orthogonal; }
    interpolationSchemes { default linear; }
    snGradSchemes { default orthogonal; }
    """

    fvSolution_content = """
    FoamFile { format ascii; class dictionary; location "system"; object fvSolution; }
    solvers {
        p { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.1; }
        U { solver PBiCGStab; preconditioner DILU; tolerance 1e-08; relTol 0; }
    }
    SIMPLE { nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
    relaxationFactors { fields { p 0.3; } equations { U 0.7; } }
    """

    transportProperties_content = """
    FoamFile { format ascii; class dictionary; location "constant"; object transportProperties; }
    transportModel Newtonian;
    nu [0 2 -1 0 0 0 0] 0.01;
    """

    turbulenceProperties_content = """
    FoamFile { format ascii; class dictionary; location "constant"; object turbulenceProperties; }
    simulationType laminar;
    """ # <<< ADDED

    U_content = """
    FoamFile { format ascii; class volVectorField; object U; }
    dimensions [0 1 -1 0 0 0 0];
    internalField uniform (0 0 0);
    boundaryField {
        movingWall { type fixedValue; value uniform (1 0 0); }
        fixedWalls { type noSlip; }
        frontAndBack { type empty; }
    }
    """

    p_content = """
    FoamFile { format ascii; class volScalarField; object p; }
    dimensions [0 2 -2 0 0 0 0];
    internalField uniform 0;
    boundaryField {
        movingWall { type zeroGradient; }
        fixedWalls { type zeroGradient; }
        frontAndBack { type empty; }
    }
    """

    # --- Write files ---
    files = {
        "system/blockMeshDict": blockMeshDict_content,
        "system/controlDict": controlDict_content,
        "system/fvSchemes": fvSchemes_content,
        "system/fvSolution": fvSolution_content,
        "constant/transportProperties": transportProperties_content,
        "constant/turbulenceProperties": turbulenceProperties_content, # <<< ADDED
        "0/U": U_content,
        "0/p": p_content
    }

    for file_path, content in files.items():
        with open(os.path.join(case_path, file_path), 'w') as f:
            f.write(content.strip())

    print(f"OpenFOAM case created at '{case_path}'")