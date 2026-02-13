from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab
from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri



class Tessellation:
    """
    Planar tessellation based on Delaunay triangulation.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Node coordinates (x, y).

    Attributes
    ----------
    points : ndarray (N, 2)
        Node coordinates.
    n_nodes : int
        Number of nodes.
    tri : scipy.spatial.Delaunay
        Delaunay triangulation object.
    simplices : ndarray (Nt, 3)
        Triangle connectivity (node indices).
    connectivity : ndarray (Ne, 2)
        Edge connectivity (unique undirected edges).
    n_elements : int
        Number of elements (edges).
    lines : ndarray (Ne, 2, 2)
        Node coordinates of each edge.
    element_length : ndarray (Ne,)
        Length of each edge.
    node_id : list[int]
        Node IDs (1-based).
    element_id : list[int]
        Element IDs (1-based).
    """

    def __init__(self, points: np.ndarray, domain_filter=None):

        # -----------------------------
        # Input validation
        # -----------------------------
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an array of shape (N, 2)")

        self.points = points
        self.n_nodes = points.shape[0]

        # -----------------------------
        # Delaunay triangulation
        # -----------------------------
        self.tri = Delaunay(points)
        self.simplices = self.tri.simplices  # (Nt, 3)

        # --- FILTRO DE DOMINIO ---
        if domain_filter is not None:
            centroids = np.mean(points[self.simplices], axis=1)
            mask = np.array([domain_filter(c) for c in centroids])
            self.simplices = self.simplices[mask]

        # -----------------------------
        # Edge extraction (unique edges)
        # -----------------------------
        edges = set()

        for tri_nodes in self.simplices:
            i, j, k = tri_nodes
            edges.add(tuple(sorted((i, j))))
            edges.add(tuple(sorted((j, k))))
            edges.add(tuple(sorted((k, i))))

        self.connectivity = np.array(list(edges), dtype=int)
        self.n_elements = self.connectivity.shape[0]

        # -----------------------------
        # Edge geometry
        # -----------------------------
        self.lines = self.points[self.connectivity]  # (Ne, 2, 2)

        diff = self.lines[:, 0, :] - self.lines[:, 1, :]
        self.element_length = np.linalg.norm(diff, axis=1)

        # -----------------------------
        # Identifiers (1-based)
        # -----------------------------
        self.node_id = np.arange(0, self.n_nodes).tolist()
        self.element_id = np.arange(0, self.n_elements).tolist()

    def plot(
        self,
        show_nodes=True,
        show_node_ids=True,
        show_elements=True,
        show_element_ids=False,
        element_ids_size = 9,
        node_ids_size = 9,
        figure_size = (6,6),
        line_color="k",
        line_width=1.0,
        node_color="r",
        node_size=30,
        ax=None
    ):
        """
        Plot the tessellation mesh.

        Parameters
        ----------
        show_nodes : bool
            Plot nodes.
        show_node_ids : bool
            Show node numbering.
        show_elements : bool
            Plot edges.
        show_element_ids : bool
            Show edge numbering.
        line_color : str
            Color of edges.
        line_width : float
            Line width of edges.
        node_color : str
            Color of nodes.
        node_size : float
            Marker size for nodes.
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, a new figure is created.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)

        # -----------------------------
        # Plot elements (edges)
        # -----------------------------
        if show_elements:
            for e, (i, j) in enumerate(self.connectivity):
                x = [self.points[i, 0], self.points[j, 0]]
                y = [self.points[i, 1], self.points[j, 1]]

                ax.plot(
                    x, y,
                    color=line_color,
                    linewidth=line_width
                )

                if show_element_ids:
                    xm = 0.5 * (x[0] + x[1])
                    ym = 0.5 * (y[0] + y[1])
                    ax.text(
                        xm, ym,
                        f"{self.element_id[e]}",
                        color=line_color,
                        fontsize=element_ids_size,
                        ha="center",
                        va="center"
                    )

        # -----------------------------
        # Plot nodes
        # -----------------------------
        if show_nodes:
            ax.scatter(
                self.points[:, 0],
                self.points[:, 1],
                c=node_color,
                s=node_size,
                zorder=3
            )

            if show_node_ids:
                for i, (x, y) in enumerate(self.points):
                    ax.text(
                        x, y,
                        f"{self.node_id[i]}",
                        color=node_color,
                        fontsize=node_ids_size,
                        ha="right",
                        va="bottom"
                    )

        # -----------------------------
        # Formatting
        # -----------------------------
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(None)
        ax.grid(False)

        plt.show()



class FEMProblem:

    def __init__(self, elements):
        """
        elements : list of element objects
        """
        self.elements = elements

        # Número total de nodos globales (deducido)
        all_nodes = np.concatenate([elem.node_ids for elem in elements])
        self.n_nodes = int(all_nodes.max() + 1)

        self.ndof = 2 * self.n_nodes

        self.K = lil_matrix((self.ndof, self.ndof))
        self.f = np.zeros(self.ndof)
        self.fixed_dofs = {}   # inicialización
        

    def apply_point_loads(self, nodes, values):
        """
        Apply point loads using vectorized operations.
        """
        nodes = np.asarray(nodes, dtype=int)
        values = np.asarray(values, dtype=float)  # (N, 2)
    
        dofs = np.column_stack((2 * nodes, 2 * nodes + 1)).ravel()
        forces = values.ravel()
    
        np.add.at(self.f, dofs, forces)
        

    def assemble_stiffness(self):
        """
        Assemble global stiffness matrix using vectorized indexing.
        """
        for elem in self.elements:
            Ke = elem.stiffness_matrix()          # (6, 6)
            node_ids = elem.node_ids              # (3,)
    
            dofs = np.repeat(2 * node_ids, 2) + np.tile([0, 1], 3)
            # dofs = [2*i,2*i+1,2*j,2*j+1,2*k,2*k+1]
    
            I, J = np.meshgrid(dofs, dofs, indexing="ij")
            self.K[I, J] += Ke


    def assemble_stiffness_v(self):
        """
        Ensambla la matriz de rigidez global de forma totalmente vectorizada.
        """
        n_elem = len(self.elements)
        if n_elem == 0:
            return
    
        # Recolectar matrices Ke y DOFs de cada elemento
        Ke_list = []
        dofs_list = []
        for elem in self.elements:
            Ke_list.append(elem.stiffness_matrix())          # (6,6)
            node_ids = elem.node_ids
            dofs = np.repeat(2 * node_ids, 2) + np.tile([0, 1], 3)  # (6,)
            dofs_list.append(dofs)
    
        # Convertir a arrays NumPy
        Ke_all = np.array(Ke_list)          # (n_elem, 6, 6)
        dofs_all = np.array(dofs_list)      # (n_elem, 6)
    
        # Generar índices de fila y columna para todas las combinaciones (i,j) por elemento
        # Para cada elemento, queremos: filas = repetir cada DOF 6 veces, columnas = repetir el vector de DOFs 6 veces
        I = np.repeat(dofs_all, 6, axis=1).ravel()    # (n_elem * 36,)
        J = np.tile(dofs_all, (1, 6)).ravel()         # (n_elem * 36,)
        V = Ke_all.ravel()                             # (n_elem * 36,)
    
        # Construir matriz en formato COO (suma duplicados automáticamente al convertir a CSR)
        K_coo = coo_matrix((V, (I, J)), shape=(self.ndof, self.ndof))
        self.K = K_coo.tocsr()

    def apply_dirichlet_bcs(self, nodes, values):
        nodes = np.asarray(nodes, dtype=int)
        values = np.asarray(values, dtype=float)
        dofs = np.column_stack((2 * nodes, 2 * nodes + 1)).ravel()
        prescribed = values.ravel()
        self.fixed_dofs.update(zip(dofs, prescribed))
        

    def apply_dirichlet_bcs_dof(self, nodes, values):
        nodes = np.asarray(nodes, dtype=int)
        values = np.asarray(values, dtype=float)
        dofs = np.column_stack((2 * nodes, 2 * nodes + 1)).ravel()
        prescribed = values.ravel()
        mask = ~np.isnan(prescribed)
        self.fixed_dofs.update(zip(dofs[mask], prescribed[mask]))

    
    def apply_distributed_loads(self, loads, nodes):
        """
        Apply distributed loads using vectorized segment operations.
    
        Parameters
        ----------
        loads : list of dict
        nodes : ndarray (N, 2)
            Global nodal coordinates
        """
        nodes = np.asarray(nodes, dtype=float)
    
        for load in loads:
            node_ids = np.asarray(load["nodes"], dtype=int)
            q = load["value"]
            angle = np.deg2rad(load["angle"])
    
            t = np.array([np.cos(angle), np.sin(angle)])
            tx, ty = q * t
    
            coords = nodes[node_ids]
            seg = coords[1:] - coords[:-1]          # (nseg, 2)
            L = np.linalg.norm(seg, axis=1)          # (nseg,)
    
            fseg = 0.5 * L[:, None] * np.array([tx, ty])  # (nseg, 2)
    
            dofs_a = np.column_stack((2 * node_ids[:-1], 2 * node_ids[:-1] + 1)).ravel()
            dofs_b = np.column_stack((2 * node_ids[1:],  2 * node_ids[1:]  + 1)).ravel()
    
            #forces = np.repeat(fseg, 2, axis=1).ravel()
            forces = fseg.ravel()
    
            np.add.at(self.f, dofs_a, forces)
            np.add.at(self.f, dofs_b, forces)


    def apply_body_force(self, bx, by, element_ids=None):
        """
        Apply body force (force per unit volume) to elements.
        For CST elements, the body force is lumped equally to the three nodes.
        Resulting nodal forces: f_i = (bx * V / 3, by * V / 3) for each node,
        where V = thickness * area.
    
        Parameters
        ----------
        bx, by : float
            Components of body force vector (N/m³).
        element_ids : array-like, optional
            Indices of elements to apply the load. If None, apply to all elements.
        """
        if element_ids is None:
            element_ids = range(len(self.elements))
    
        for idx in element_ids:
            elem = self.elements[idx]
    
            # Volumen del elemento (espesor * área)
            if not hasattr(elem, 'area') or not hasattr(elem, 't'):
                raise AttributeError("Element must have attributes 'area' and 't' (thickness).")
            vol = elem.t * elem.area
    
            # Vector de fuerzas nodales equivalentes (6 componentes: fx1, fy1, fx2, fy2, fx3, fy3)
            fe = (vol / 3.0) * np.array([bx, by, bx, by, bx, by])
    
            # Grados de libertad del elemento
            dofs = np.repeat(2 * elem.node_ids, 2) + np.tile([0, 1], 3)
    
            # Acumular en el vector global
            np.add.at(self.f, dofs, fe)


    def _init_fixed_dofs(self):
        """Inicializa el diccionario de DOFs fijos si no existe."""
        if not hasattr(self, "fixed_dofs"):
            self.fixed_dofs = {}
    
    def apply_dirichlet_on_ux(self, nodes, values):
        """
        Fija el desplazamiento en X (grado de libertad 2*node) para los nodos indicados.
    
        Parameters
        ----------
        nodes : array-like of int
            IDs de los nodos.
        values : float or array-like
            Valor prescrito de ux. Si es escalar, se aplica a todos los nodos.
        """
        self._apply_dirichlet_on_dof(nodes, dof=0, values=values)
    
    def apply_dirichlet_on_uy(self, nodes, values):
        """
        Fija el desplazamiento en Y (grado de libertad 2*node+1) para los nodos indicados.
        """
        self._apply_dirichlet_on_dof(nodes, dof=1, values=values)
    
    def apply_dirichlet_on_dof(self, nodes, dof, values):
        """
        Método genérico para fijar un grado de libertad específico por nodo.
    
        Parameters
        ----------
        nodes : array-like of int
            IDs de los nodos.
        dof : int (0 o 1)
            0 → desplazamiento en X, 1 → desplazamiento en Y.
        values : float or array-like
            Valor(es) prescrito(s).
        """
        self._apply_dirichlet_on_dof(nodes, dof, values)
    
    def _apply_dirichlet_on_dof(self, nodes, dof, values):
        """
        Lógica interna: actualiza self.fixed_dofs de forma acumulativa.
        """
        self._init_fixed_dofs()
    
        nodes = np.asarray(nodes, dtype=int)
        values = np.asarray(values, dtype=float)
    
        # Si values es escalar, se expande a un array de la misma longitud que nodes
        if values.ndim == 0:
            values = np.full(len(nodes), values)
    
        # DOFs globales correspondientes (solo el grado seleccionado)
        global_dofs = 2 * nodes + dof
    
        # Actualizar el diccionario (sobrescribe si el DOF ya estaba fijado)
        for dof_id, val in zip(global_dofs, values):
            self.fixed_dofs[dof_id] = val
            


    def partition_system(self):
        """
        Partition the global system into free and constrained DOFs.

        Returns
        -------
        dict with keys:
            K_ff, K_fc, f_f, u_c, free_dofs, fixed_dofs
        """
        if not hasattr(self, "fixed_dofs"):
            raise RuntimeError("Dirichlet BCs have not been applied.")

        fixed_dofs = np.array(sorted(self.fixed_dofs.keys()), dtype=int)
        u_c = np.array([self.fixed_dofs[d] for d in fixed_dofs])

        all_dofs = np.arange(self.ndof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        # Convert once to CSR for efficient slicing
        K = self.K.tocsr()

        K_ff = K[free_dofs][:, free_dofs]
        K_fc = K[free_dofs][:, fixed_dofs]

        f_f = self.f[free_dofs] - K_fc @ u_c

        return {
            "K_ff": K_ff,
            "f_f": f_f,
            "u_c": u_c,
            "free_dofs": free_dofs,
            "fixed_dofs": fixed_dofs
        }


    def solve(self, method="direct", tol=1e-8, maxiter=1000):
        """
        Solve the FEM system using the selected solver.

        Parameters
        ----------
        method : str
            'direct', 'cg', 'gmres', 'bicgstab'
        tol : float
            Solver tolerance
        maxiter : int
            Maximum iterations (Krylov methods)

        Returns
        -------
        u_f : ndarray
            Displacements at free DOFs
        """
        data = self.partition_system()
        K_ff = data["K_ff"]
        f_f = data["f_f"]

        if method == "direct":
            u_f = spsolve(K_ff, f_f)

        elif method == "cg":
            u_f, info = cg(K_ff, f_f, rtol=tol, maxiter=maxiter)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info={info})")

        elif method == "gmres":
            u_f, info = gmres(K_ff, f_f, tol=tol, maxiter=maxiter)
            if info != 0:
                raise RuntimeError(f"GMRES did not converge (info={info})")

        elif method == "bicgstab":
            u_f, info = bicgstab(K_ff, f_f, tol=tol, maxiter=maxiter)
            if info != 0:
                raise RuntimeError(f"BiCGSTAB did not converge (info={info})")

        else:
            raise ValueError(f"Unknown solver method '{method}'")

        self.u_f = u_f
        self._partition_data = data

        x = np.zeros(self.ndof)
        x[data["free_dofs"]] = self.u_f
        x[data["fixed_dofs"]] = data["u_c"]

        self.ux = x[0::2]
        self.uy = x[1::2]
        self.umag = np.sqrt(self.ux**2 + self.uy**2)
        
        self.x = x
        return 

    def plot_stiffness_sparsity(self, partitioned=False, figsize=(6, 6)):
        """
        Visualize sparsity pattern of the global stiffness matrix.
    
        Parameters
        ----------
        partitioned : bool
            If True, plots only K_ff (free-free block).
            If False, plots full global matrix K.
        figsize : tuple
            Figure size.
        """
    
        if partitioned:
            data = self.partition_system()
            Kplot = data["K_ff"]
            title = "Sparsity pattern of K_ff (partitioned)"
        else:
            Kplot = self.K.tocsr()
            title = "Sparsity pattern of global K"
    
        plt.figure(figsize=figsize)
        plt.spy(Kplot, markersize=1)
        plt.title(title)
        plt.xlabel("DOF")
        plt.ylabel("DOF")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.show()

    

    def compute_element_stresses(self):
        """
        Compute stresses for each element (CST).
    
        Returns
        -------
        sigma : ndarray (n_elements, 3)
            Stress vector per element:
            [sigma_x, sigma_y, tau_xy]
        """
    
        if not hasattr(self, "x"):
            raise RuntimeError("System must be solved before computing stresses.")
    
        sigma = []
    
        for elem in self.elements:
    
            node_ids = elem.node_ids
    
            # DOFs del elemento
            dofs = np.repeat(2 * node_ids, 2) + np.tile([0, 1], 3)
    
            Ue = self.x[dofs]  # (6,)
    
            # deformación
            eps = elem.B @ Ue  # (3,)
    
            # tensión
            s = elem.D @ eps   # (3,)
    
            sigma.append(s)
    
        sigma = np.array(sigma)
    
        self.element_stress = sigma

        sigma_xx = sigma[:, 0]
        sigma_yy = sigma[:, 1]
        tau_xy   = sigma[:, 2]
        sigma_vm = np.sqrt(
        sigma_xx**2 - sigma_xx * sigma_yy + sigma_yy**2 + 3 * tau_xy**2
        )
        # guardar como atributos
        self.sigma_xx = sigma_xx
        self.sigma_yy = sigma_yy
        self.tau_xy   = tau_xy
        self.sigma_vm = sigma_vm
        return 

    def plot_elemental_variable(
        self,
        nodes,
        simplices,
        elemental_values,
        title="Elemental variable",
        cmap="jet",
        show_mesh=False,
        show_scale=False
    ):
        """
        Plot scalar field defined per element (CST constant value).
    
        Parameters
        ----------
        nodes : ndarray (N,2)
            Global nodal coordinates
        simplices : ndarray (Nt,3)
            Triangle connectivity
        elemental_values : ndarray (Nt,)
            One scalar value per triangle
        """
    
        nodes = np.asarray(nodes)
        simplices = np.asarray(simplices)
        elemental_values = np.asarray(elemental_values)
    
        if elemental_values.shape[0] != simplices.shape[0]:
            raise ValueError("elemental_values must have one value per triangle")
    
        triang = mtri.Triangulation(
            nodes[:, 0],
            nodes[:, 1],
            simplices
        )
    
        plt.figure(figsize=(6, 5))
    
        # COLOR POR ELEMENTO
        tpc = plt.tripcolor(
            triang,
            facecolors=elemental_values,
            cmap=cmap,
            edgecolors='k' if show_mesh else 'none'
        )
    
        plt.colorbar(tpc)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.gca().set_aspect("equal")
    
        if not show_scale:
            plt.axis("off")
    
        plt.tight_layout()
        plt.show()



class TriangularElement:
    """
    Linear triangular finite element (CST) for plane stress elasticity.
    """

    def __init__(
        self,
        node_ids,
        nodes,
        E,
        nu,
        thickness=1.0,
        element_type="CST"
    ):
        """
        Parameters
        ----------
        node_ids : list[int] or ndarray (3,)
            Global node indices [i, j, k]
        nodes : ndarray (N, 2)
            Global nodal coordinates
        """
        self.node_ids = np.asarray(node_ids, dtype=int)

        if self.node_ids.shape != (3,):
            raise ValueError("node_ids must have length 3")

        self.coords = nodes[self.node_ids]

        self.E = E
        self.nu = nu
        self.t = thickness
        self.type = element_type

        self._compute_geometry()
        self._compute_material_matrix()
        self._compute_B_matrix()


    def _compute_geometry(self):
        xi, yi = self.coords[0]
        xj, yj = self.coords[1]
        xk, yk = self.coords[2]

        # Área
        self.area = 0.5 * np.linalg.det(np.array([
            [1, xi, yi],
            [1, xj, yj],
            [1, xk, yk]
        ]))

        if self.area <= 0:
            raise ValueError("Element has zero or negative area")

        # Coeficientes geométricos
        self.alpha = np.array([
            xj * yk - xk * yj,
            xk * yi - xi * yk,
            xi * yj - xj * yi
        ])

        self.beta = np.array([
            yj - yk,
            yk - yi,
            yi - yj
        ])

        self.delta = np.array([
            xk - xj,
            xi - xk,
            xj - xi
        ])


    def _compute_material_matrix(self):
        E = self.E
        nu = self.nu

        self.D = (E / (1.0 - nu**2)) * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])


    def _compute_B_matrix(self):
        A2 = 2.0 * self.area
        b = self.beta
        d = self.delta

        self.B = (1.0 / A2) * np.array([
            [ b[0],    0.0,  b[1],    0.0,  b[2],    0.0 ],
            [ 0.0,   d[0],  0.0,   d[1],  0.0,   d[2] ],
            [ d[0],  b[0],  d[1],  b[1],  d[2],  b[2] ]
        ])

    def stiffness_matrix(self):
        """
        Returns
        -------
        Ke : ndarray (6,6)
            Element stiffness matrix
        """
        Ke = self.t * self.area * (self.B.T @ self.D @ self.B)
        return Ke


    def strain_energy(self, Ue):
        """
        Element strain energy.

        Parameters
        ----------
        Ue : ndarray (6,)
            Element nodal displacement vector

        Returns
        -------
        float
        """
        Ke = self.stiffness_matrix()
        return 0.5 * Ue.T @ Ke @ Ue


def plot_fem_contour(points, simplices, nodal_values,
                     title = "Contour plot",
                     cmap = "jet",
                     levels = 30,
                     show_scale = False,
                     show_mesh=False):
    """
    Plot FEM contour map on triangular mesh.

    Parameters
    ----------
    points : ndarray (N,2)
    simplices : ndarray (Nt,3)
    nodal_values : ndarray (N,)
    title : str
    cmap : str
    levels : int
    show_mesh : bool
    """

    points = np.asarray(points)
    nodal_values = np.asarray(nodal_values)

    triang = mtri.Triangulation(points[:, 0], points[:, 1], simplices)

    plt.figure(figsize=(6, 5))

    contour = plt.tricontourf(
        triang,
        nodal_values,
        levels=levels,
        cmap=cmap
    )

    if show_mesh:
        plt.triplot(triang, color='k', linewidth=0.4, alpha=0.5)

    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    if not show_scale:
        plt.axis("off")
    plt.tight_layout()
    plt.show()



def plot_deformed_contour(points, simplices, ux, uy,
                          scale=1.2,
                          cmap="turbo",
                          levels=50,
                          show_scale = False,
                          title = "Deformed shape (scaled)"):

    deformed = points + scale * np.column_stack((ux, uy))

    triang = mtri.Triangulation(
        deformed[:, 0],
        deformed[:, 1],
        simplices
    )

    umag = np.sqrt(ux**2 + uy**2)

    plt.figure(figsize=(6,5))
    contour = plt.tricontourf(
        triang,
        umag,
        levels=levels,
        cmap=cmap
    )

    plt.colorbar(contour)
    plt.gca().set_aspect("equal")
    plt.title(title)
    if not show_scale:
        plt.axis("off")
    plt.tight_layout()
    plt.show()

