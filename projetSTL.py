import marimo

__generated_with = "0.9.17"
app = marimo.App()


@app.cell
def __(mo):
    mo.md("""#3D Geometry File Formats""")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## About STL

        STL is a simple file format which describes 3D objects as a collection of triangles.
        The acronym STL stands for "Simple Triangle Language", "Standard Tesselation Language" or "STereoLitography"[^1].

        [^1]: STL was invented for – and is still widely used – for 3D printing.
        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


@app.cell
def __(mo):
    with open("data/teapot.stl", mode="rt", encoding="utf-8") as _file:
        teapot_stl = _file.read()

    teapot_stl_excerpt = teapot_stl[:723] + "..." + teapot_stl[-366:]

    mo.md(
        f"""
    ## STL ASCII Format

    The `data/teapot.stl` file provides an example of the STL ASCII format. It is quite large (more than 60000 lines) and looks like that:
    """
    +
    f"""```
    {teapot_stl_excerpt}
    ```
    """
    +

    """
    """
    )
    return teapot_stl, teapot_stl_excerpt


@app.cell
def __(mo):
    mo.md(f"""

      - Study the [{mo.icon("mdi:wikipedia")} STL (file format)](https://en.wikipedia.org/wiki/STL_(file_format)) page (or other online references) to become familiar the format.

      - Create a STL ASCII file `"data/cube.stl"` that represents a cube of unit length  
        (💡 in the simplest version, you will need 12 different facets).

      - Display the result with the function `show` (make sure to check different angles).
    """)
    return


@app.cell
def __(mo, np, show):
    sommets = [
        (0, 0, 0),  # 0
        (1, 0, 0),   # 1
        (1, 1, 0),    # 2
        (0, 1, 0),   # 3
        (0, 0, 1),   # 4
        (1, 0, 1),    # 5
        (1, 1, 1),     # 6
        (0, 1, 1),    # 7
    ]
    faces = [
          # Face bas
        (0, 2, 3),(0, 1, 2), (0, 1, 5), (0, 3, 7),(1, 2, 6),(2, 3, 7),
        (0, 5, 4),  (0, 7, 4), (1, 6, 5),
         (2, 7, 6),(4, 5, 6),(4, 6, 7)  # Face haut
    ]
    with open('data/cube.stl', "w") as file:
            file.write("solid cube\n")
            for face in faces:
                vecteur1=np.array(sommets[face[1]])-np.array(sommets[face[0]]) #on définit deux vecteurs 'appartenant'au plan
                vecteur2=np.array(sommets[face[2]])-np.array(sommets[face[0]])
                normalspasnorme=np.cross(vecteur1,vecteur2)#le pdt vect est forcément 
                normals=normalspasnorme/np.sqrt(normalspasnorme[0]**2+normalspasnorme[1]**2+normalspasnorme[2]**2)
                file.write(f"  facet normal {normals[0]} {normals[1]} {normals[2]} \n")  
                file.write("    outer loop\n")
                for vertex_index in face:
                    vertex = sommets[vertex_index]
                    file.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
                file.write("    endloop\n")
                file.write("  endfacet\n")
            file.write("endsolid cube\n")

    mo.show_code(show("data/cube.stl", theta=45.0, phi=45.0, scale=0.5))
    return (
        face,
        faces,
        file,
        normals,
        normalspasnorme,
        sommets,
        vecteur1,
        vecteur2,
        vertex,
        vertex_index,
    )


@app.cell
def __(mo):
    mo.md(r"""## STL & NumPy""")
    return


@app.cell
def __(np):
    def make_STL(triangles, normals=None, name=""):
        STL=f'solid {name}\n'
        for triangle in triangles:
            vecteur1=triangle[1]-triangle[0] #on définit deux vecteurs 'appartenant'au plan
            vecteur2=triangle[2]-triangle[0]
            normalspasnorme=np.cross(vecteur1,vecteur2)#le pdt vect est forcément 
            normals=normalspasnorme/np.sqrt(normalspasnorme[0]**2+normalspasnorme[1]**2+normalspasnorme[2]**2)
            STL+=f"  facet normal {normals[0]} {normals[1]} {normals[2]}\n" 
            STL+="    outer loop\n"
            for sommet in triangle:
                STL+=f"      vertex {sommet[0]} {sommet[1]} {sommet[2]}\n"
            STL+="    endloop\n"
            STL+="  endfacet\n"
        STL+=f"endsolid {name}\n"
        return STL
    return (make_STL,)


@app.cell
def __(make_STL, np):
    make_STL(triangles= np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    ),name='dodo') #test
    return


@app.cell
def __(mo):
    mo.md(rf"""

    ### NumPy to STL

    Implement the following function:

    ```python
    def make_STL(triangles, normals=None, name=""):
        pass # 🚧 TODO!
    ```

    #### Parameters

      - `triangles` is a NumPy array of shape `(n, 3, 3)` and data type `np.float32`,
         which represents a sequence of `n` triangles (`triangles[i, j, k]` represents 
         is the `k`th coordinate of the `j`th point of the `i`th triangle)

      - `normals` is a NumPy array of shape `(n, 3)` and data type `np.float32`;
         `normals[i]` represents the outer unit normal to the `i`th facet.
         If `normals` is not specified, it should be computed from `triangles` using the 
         [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

      - `name` is the (optional) solid name embedded in the STL ASCII file.

    #### Returns

      - The STL ASCII description of the solid as a string.

    #### Example

    Given the two triangles that make up a flat square:

    ```python

    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    ```

    then printing `make_STL(square_triangles, name="square")` yields
    ```
    solid square
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 0.0 0.0 0.0
          vertex 1.0 0.0 0.0
          vertex 0.0 1.0 0.0
        endloop
      endfacet
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 1.0 1.0 0.0
          vertex 0.0 1.0 0.0
          vertex 1.0 0.0 0.0
        endloop
      endfacet
    endsolid square
    ```

    """)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### STL to NumPy

        Implement a `tokenize` function


        ```python
        def tokenize(stl):
            tokens=stl.split(' ')
            return (tokens)
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `stl`: a Python string that represents a STL ASCII model.

        #### Returns

          - `tokens`: a list of STL keywords (`solid`, `facet`, etc.) and `np.float32` numbers.

        #### Example

        For the ASCII representation the square `data/square.stl`, printing the tokens with

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        print(tokens)
        ```

        yields

        ```python
        ['solid', 'square', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(0.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'endloop', 'endfacet', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(1.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'endloop', 'endfacet', 'endsolid', 'square']
        ```
        """
    )
    return


@app.cell
def __(np):
    def tokenize(stl):
        tokens=stl.split()
        for i,item in enumerate(tokens):
            try:
                tokens[i]=np.float32(item)
            except:
                continue
        return (tokens)
    return (tokenize,)


@app.cell
def __(make_STL, np, tokenize):
    tokenize(make_STL(triangles= np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32),name='test'
    ))
    return


@app.cell
def __(mo):
    mo.md(
        """
        Implement a `parse` function


        ```python
        def parse(tokens):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `tokens`: a list of tokens

        #### Returns

        A `triangles, normals, name` triple where

          - `triangles`: a `(n, 3, 3)` NumPy array with data type `np.float32`,

          - `normals`: a `(n, 3)` NumPy array with data type `np.float32`,

          - `name`: a Python string.

        #### Example

        For the ASCII representation `square_stl` of the square,
        tokenizing then parsing

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        triangles, normals, name = parse(tokens)
        print(repr(triangles))
        print(repr(normals))
        print(repr(name))
        ```

        yields

        ```python
        array([[[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.]],

               [[1., 1., 0.],
                [0., 1., 0.],
                [1., 0., 0.]]], dtype=float32)
        array([[0., 0., 1.],
               [0., 0., 1.]], dtype=float32)
        'square'
        ```
        """
    )
    return


@app.cell
def __(np):
    def parse (tokens):
        name=tokens[-1]
        n=0
        for val in tokens:#compte le nombre de triangles nécessaire
            if val=='facet':
                n+=1
        normals=np.empty((n,3))
        triangle=np.empty((n,3,3))
        tr=-1
        for i,item in enumerate(tokens):
            if item=='facet':
                tr+=1
            if item=='normal':
                normals[tr]=np.array([tokens[i+1],tokens[i+2],tokens[i+3]])
            if item=='loop':
                triangle[tr]=[[tokens[i+2],tokens[i+3],tokens[i+4]],[tokens[i+6],tokens[i+7],tokens[i+8]],[tokens[i+10],tokens[i+11],tokens[i+12]]]
        return(triangle,normals,name)
    return (parse,)


@app.cell
def __(make_STL, np, parse, tokenize):
    parse(tokenize(make_STL(triangles= np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32
    ),name='marche-stp') ))            #test
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Rules & Diagnostics



        Make diagnostic functions that check whether a STL model satisfies the following rules

          - **Positive octant rule.** All vertex coordinates are non-negative.

          - **Orientation rule.** All normals are (approximately) unit vectors and follow the [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

          - **Shared edge rule.** Each triangle edge appears exactly twice.

          - **Ascending rule.** the z-coordinates of (the barycenter of) each triangle are a non-decreasing sequence.

    When the rule is broken, make sure to display some sensible quantitative measure of the violation (in %).

    For the record, the `data/teapot.STL` file:

      - 🔴 does not obey the positive octant rule,
      - 🟠 almost obeys the orientation rule, 
      - 🟢 obeys the shared edge rule,
      - 🔴 does not obey the ascending rule.

    Check that your `data/cube.stl` file does follow all these rules, or modify it accordingly!

    """
    )
    return


@app.cell
def __(np, parse, tokenize):
    def verif(stl):
        triangle,normals,name=parse(tokenize(stl))
        n=len(normals)
        nb_neg=0
        for valeur in np.nditer(triangle):
            if valeur<0:
                nb_neg+=1
        print(f'{nb_neg} valeurs sont négatives soient{100*(nb_neg)/(9*n)}%')
        z=triangle[0][0][2]+triangle[0][1][2]+triangle[0][2][2]
        errors=0
        for elt in triangle:
            newz=elt[0][2]+elt[1][2]+elt[2][2]
            if newz<z:
                errors+=1
            z=newz
        print(f'il y a approximativement {errors} erreurs d_ascendance soit environ {100*errors/n}%' )
        echecnorme=0
        for elem in normals:
            norme=np.sqrt(elem[0]**2+elem[1]**2+elem[2]**2)
            if norme<0.99 or norme>1.01:
                echecnorme+=1
        print(f'il y a {echecnorme} erreurs de norme soit environ {100*echecnorme/n}%' )
        apparitions={}
        for facette in triangle:
            for k in range (3):
                a=np.concatenate((facette[k],facette[(k+1)%3]))
                a1=tuple(a)
                if a1 in apparitions:
                    apparitions[a1]+=1
                b=np.concatenate((facette[(k+1)%3],facette[k]))
                b1=tuple(b)
                if b1 in apparitions:
                    apparitions[b1]+=1
                if b1 not in apparitions and a1 not in apparitions:
                    apparitions[a1]=1
        erreur=0
        for nb in apparitions.values():
            if nb !=2:
                erreur+=1
        print(f'il y a {erreur} erreurs de "shared edge rule" soit environ {100*erreur/(1.5*n)}%' )
    return (verif,)


@app.cell
def __(verif):
    with open("data/cube.stl", mode="rt", encoding="us-ascii") as cube_file:
        cube_stl = cube_file.read()
    verif(cube_stl)
    return cube_file, cube_stl


@app.cell
def __(mo):
    mo.md(
    rf"""
    ## OBJ Format

    The OBJ format is an alternative to the STL format that looks like this:

    ```
    # OBJ file format with ext .obj
    # vertex count = 2503
    # face count = 4968
    v -3.4101800e-003 1.3031957e-001 2.1754370e-002
    v -8.1719160e-002 1.5250145e-001 2.9656090e-002
    v -3.0543480e-002 1.2477885e-001 1.0983400e-003
    v -2.4901590e-002 1.1211138e-001 3.7560240e-002
    v -1.8405680e-002 1.7843055e-001 -2.4219580e-002
    ...
    f 2187 2188 2194
    f 2308 2315 2300
    f 2407 2375 2362
    f 2443 2420 2503
    f 2420 2411 2503
    ```

    This content is an excerpt from the `data/bunny.obj` file.

    """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/bunny.obj", scale="1.5"))
    return


@app.cell
def __(mo):
    mo.md(
        """
        Study the specification of the OBJ format (search for suitable sources online),
        then develop a `OBJ_to_STL` function that is rich enough to convert the OBJ bunny file into a STL bunny file.
        """
    )
    return


@app.cell
def __(make_STL, np, tokenize):
    def OBJ_to_STL(lien):
        with open(lien, mode="rt", encoding="us-ascii") as file:
            obj = file.read()
        tokens = tokenize(obj)
        for i,item in enumerate(tokens):
            if item=='vertex':
                nb_v=int(tokens[i+3])
            if item=='face':
                nb_f=int(tokens[i+3])
        sommets=np.zeros((nb_v,3))
        faces=np.zeros((nb_f,3,3))
        sommet=0
        face=0
        for k,item in enumerate(tokens):
            if item=='v':
                sommets[sommet]=[tokens[k+1],tokens[k+2],tokens[k+3]]
                sommet+=1
            if item=='f':
                faces[face]=[sommets[int(tokens[k+1])-1],sommets[int(tokens[k+2])-1],sommets[int(tokens[k+3])-1]]
                face+=1
        with open('data/bunny.stl', "w") as file:
            file.write(make_STL(faces))
    return (OBJ_to_STL,)


@app.cell
def __(OBJ_to_STL):
    OBJ_to_STL("data/bunny.obj")
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/bunny.stl", theta=180, phi=180, scale=1.5))
    return


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Binary STL

    Since the STL ASCII format can lead to very large files when there is a large number of facets, there is an alternate, binary version of the STL format which is more compact.

    Read about this variant online, then implement the function

    ```python
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        pass  # 🚧 TODO!
    ```

    that will convert a binary STL file to a ASCII STL file. Make sure that your function works with the binary `data/dragon.stl` file which is an example of STL binary format.

    💡 The `np.fromfile` function may come in handy.

        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/dragon.stl", theta=75.0, phi=-20.0, scale=1.7))
    return


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
            normals = []
            faces = []
            for i in range(n):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        stl_text = make_STL(faces, normals)
        with open(stl_filename_out, mode="wt", encoding="utf-8") as file:
            file.write(stl_text)
    return (STL_binary_to_text,)


@app.cell
def __(mo):
    mo.md(rf"""## Constructive Solid Geometry (CSG)

    Have a look at the documentation of [{mo.icon("mdi:github")}fogleman/sdf](https://github.com/fogleman/) and study the basics. At the very least, make sure that you understand what the code below does:
    """)
    return


@app.cell
def __(X, Y, Z, box, cylinder, mo, show, sphere):
    demo_csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    demo_csg = demo_csg - (_c.orient(X) | _c.orient(Y) | _c.orient(Z))
    demo_csg.save('output/demo-csg.stl', step=0.05)
    mo.show_code(show("output/demo-csg.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg,)


@app.cell
def __(mo):
    mo.md("""ℹ️ **Remark.** The same result can be achieved in a more procedural style, with:""")
    return


@app.cell
def __(
    box,
    cylinder,
    difference,
    intersection,
    mo,
    orient,
    show,
    sphere,
    union,
):
    demo_csg_alt = difference(
        intersection(
            sphere(1),
            box(1.5),
        ),
        union(
            orient(cylinder(0.5), [1.0, 0.0, 0.0]),
            orient(cylinder(0.5), [0.0, 1.0, 0.0]),
            orient(cylinder(0.5), [0.0, 0.0, 1.0]),
        ),
    )
    demo_csg_alt.save("output/demo-csg-alt.stl", step=0.05)
    mo.show_code(show("output/demo-csg-alt.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg_alt,)


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## JupyterCAD

    [JupyterCAD](https://github.com/jupytercad/JupyterCAD) is an extension of the Jupyter lab for 3D geometry modeling.

      - Use it to create a JCAD model that correspond closely to the `output/demo_csg` model;
    save it as `data/demo_jcad.jcad`.

      - Study the format used to represent JupyterCAD files (💡 you can explore the contents of the previous file, but you may need to create some simpler models to begin with).

      - When you are ready, create a `jcad_to_stl` function that understand enough of the JupyterCAD format to convert `"data/demo_jcad.jcad"` into some corresponding STL file.
    (💡 do not tesselate the JupyterCAD model by yourself, instead use the `sdf` library!)


        """
    )
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md("""## Appendix""")
    return


@app.cell
def __(mo):
    mo.md("""### Dependencies""")
    return


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera

    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference

    mo.show_code()
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        sphere,
        union,
    )


@app.cell
def __(mo):
    mo.md(r"""### STL Viewer""")
    return


@app.cell
def __(Camera, Mesh, glm, meshio, mo, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        vertices = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        vertices = glm.fit_unit_cube(vertices)
        mesh = Mesh(
            ax,
            camera.transform,
            vertices,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        return mo.center(fig)

    mo.show_code()
    return (show,)


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


if __name__ == "__main__":
    app.run()
