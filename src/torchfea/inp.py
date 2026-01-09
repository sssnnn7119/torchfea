from io import TextIOWrapper
from typing import TextIO
import numpy as np
from threading import Thread

class FEA_INP():

    class Parts():
        def __init__(self) -> None:
            self.elems: dict['str', np.ndarray]
            self.nodes: np.ndarray
            self.sections: list
            self.sets_nodes: dict['str', set]
            self.sets_elems: dict['str', set]
            self.surfaces: dict[str, list[tuple[np.ndarray, int]]]
            self.surfaces_tri: dict[str, np.ndarray]
            
            self.num_elems_3D: int
            self.num_elems_2D: int
            self.elems_material: np.ndarray
            """
            0: index of the element\n
            1: density of the element\n
            2: type of the element\n
            3-: parameter of the element
            """
            # self.sets_nodes = {}
            # self.sets_elems = {}
            # self.num_elems_3D = 0
            # self.num_elems_2D = 0
            # section = []

        def read(self, origin_data: list[str], ind: int):

            self.elems = {}
            self.sets_nodes = {}
            self.sets_elems = {}
            self.surfaces = {}
            self.num_elems_3D = 0
            self.num_elems_2D = 0
            section = []
            self.surfaces_tri = {}
            while ind < len(origin_data):
                now = origin_data[ind]
                if len(now) > 9 and now[0:9].lower() == '*end part':
                    break
                # case element
                if len(now) >= 22 and now[0:21].lower() == '*element, type=c3d10h':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D10H'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 21 and now[0:20].lower() == '*element, type=c3d10':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D10'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 21 and now[0:20].lower() == '*element, type=c3d15':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D15'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue
                
                if len(now) >= 21 and now[0:20].lower() == '*element, type=c3d20':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    
                    num_line = round(len(datalist)/2)
                    datalist = [datalist[2*i]+datalist[2*i+1] for i in range(num_line)]
                    self.elems['C3D20'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue
                
                if len(now) >= 21 and now[0:20].lower() == '*element, type=c3d8r':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D8R'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 21 and now[0:20].lower() == '*element, type=c3d4h':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D4H'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 20 and now[0:19].lower() == '*element, type=c3d4':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D4'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue
                
                if len(now) >= 20 and now[0:19].lower() == '*element, type=c3d8':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D8'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 20 and now[0:19].lower() == '*element, type=c3d6':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['C3D6'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_3D += ind1 - ind0
                    continue

                if len(now) >= 17 and now[0:17].lower() == '*element, type=s3':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        int(i) for i in row.replace('\n', '').replace(
                            ',', ' ').strip().split()
                    ] for row in origin_data[ind0:ind1]]
                    self.elems['S3'] = np.array(datalist, dtype=int) - 1
                    self.num_elems_2D += ind1 - ind0
                    continue

                # case node set
                if len(now
                    ) >= 12 and now[0:12].lower() == '*nset, nset=':
                    # name = now[12:].replace('\n', '').strip()
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        self.sets_nodes[name] = set(
                            (np.array(datalist, dtype=int) - 1).tolist())
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        self.sets_nodes[name] = set(
                            (np.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                    continue

                # case element set
                if len(now) >= 14 and now[
                        0:14].lower() == '*elset, elset=':
                    # name = now[14:].replace('\n', '').strip()
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        self.sets_elems[name] = set(
                            (np.array(datalist, dtype=int) - 1).tolist())
                        continue
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        self.sets_elems[name] = set(
                            (np.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                        continue
                
                # case surfaces
                if len(now) >= 8 and now[0:8].lower() == '*surface':
                    data_now = now.split('=')
                    ind += 1
                    if len(self.elems.keys()) == 0:
                        continue
                    if data_now[1].split(',')[0].strip()[:7].lower() == 'element':
                        name = data_now[2].strip()
                        self.surfaces[name] = []
                        surfaceList = []
                        while origin_data[ind][0] != '*':
                            data_now = origin_data[ind].split(',')
                            ind+=1
                            elem_set_name = data_now[0].strip()
                            surface_index = int(data_now[1].strip()[1:])
                            for key in list(self.elems.keys()):
                                elem_now = self.elems[key]
                                elem_index = np.where(np.isin(elem_now[:, 0],
                                                        list(self.sets_elems[elem_set_name])))[0]
                                elem = elem_now[elem_index]
                                if elem.shape[1] == 5:
                                    if surface_index == 1:
                                        surfaceList.append(elem[:, [1,3,2]])
                                    elif surface_index == 2:
                                        surfaceList.append(elem[:, [1,2,4]])
                                    elif surface_index == 3:
                                        surfaceList.append(elem[:, [2,3,4]])
                                    elif surface_index == 4:
                                        surfaceList.append(elem[:, [3,1,4]])
                                elif elem.shape[1] == 11:
                                    if surface_index == 1:
                                        surfaceList.append(elem[:, [1,7,5]])
                                        surfaceList.append(elem[:, [2,5,6]])
                                        surfaceList.append(elem[:, [3,6,7]])
                                        surfaceList.append(elem[:, [5,7,6]])
                                    elif surface_index == 2:
                                        surfaceList.append(elem[:, [1,5,8]])
                                        surfaceList.append(elem[:, [2,9,5]])
                                        surfaceList.append(elem[:, [4,8,9]])
                                        surfaceList.append(elem[:, [5,9,8]])
                                    elif surface_index == 3:
                                        surfaceList.append(elem[:, [2,6,9]])
                                        surfaceList.append(elem[:, [3,10,6]])
                                        surfaceList.append(elem[:, [4,9,10]])
                                        surfaceList.append(elem[:, [6,10,9]])
                                    elif surface_index == 4:
                                        surfaceList.append(elem[:, [1,8,7]])
                                        surfaceList.append(elem[:, [3,7,10]])
                                        surfaceList.append(elem[:, [4,10,8]])
                                        surfaceList.append(elem[:, [8,10,7]])
                                        
                                self.surfaces[name].append((elem_now[elem_index, 0], surface_index-1))
                            try:
                                self.surfaces_tri[name] = np.concatenate(surfaceList, axis=0)
                            except ValueError:
                                self.surfaces_tri[name] = np.array([])

                    continue

                # case node
                if len(now) >= 5 and now[0:5].lower() == '*node':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        float(i) for i in row.replace('\n', '').strip().split(',')
                    ] for row in origin_data[ind0:ind1]]
                    self.nodes = np.array(datalist, dtype=float)
                    self.nodes[:, 0] -= 1
                    continue

                # case section
                if len(now) >= 11 and now[0:11].lower() == '** section:':
                    name = now.split(':')[1].strip()
                    ind += 1
                    now = origin_data[ind]
                    data = now.split(',')
                    section_set = data[1].split('=')[1].strip()
                    section_material = data[2].split('=')[1].strip()
                    section.append([section_set, section_material])

                # case finished
                if len(now) >= 9 and now[0:9].lower() == '*end part':
                    break

                ind += 1

            self.sections = section
            self.elems_material = -np.ones(
                [self.num_elems_2D + self.num_elems_3D, 5], dtype=float)

        def write_inp(self, f: TextIOWrapper, part_name: str):
            """
            Write part data to INP file format.
            
            Args:
                f: File object to write to
                part_name (str): Name of the part
            """
            f.write(f"*Part, name={part_name}\n")
            
            # Write nodes
            if hasattr(self, 'nodes') and self.nodes is not None:
                f.write("*Node\n")
                for i, node in enumerate(self.nodes):
                    node_id = int(node[0].item()) + 1  # Convert back to 1-based indexing
                    coords = node[1:].tolist()
                    f.write(f"{node_id}, {coords[0]}, {coords[1]}, {coords[2]}\n")
            
            # Write elements
            if hasattr(self, 'elems'):
                for elem_type, elem_data in self.elems.items():
                    f.write(f"*Element, type={elem_type}\n")
                    
                    if elem_type == 'C3D20':
                        # C3D20 elements need special handling (two lines per element)
                        for elem in elem_data:
                            elem_id = int(elem[0].item()) + 1
                            nodes_line1 = [str(int(node.item()) + 1) for node in elem[1:11]]
                            nodes_line2 = [str(int(node.item()) + 1) for node in elem[11:21]]
                            f.write(f"{elem_id}, " + ", ".join(nodes_line1) + "\n")
                            f.write("" + ", ".join(nodes_line2) + "\n")
                    else:
                        # Standard elements (one line per element)
                        for elem in elem_data:
                            elem_id = int(elem[0].item()) + 1
                            nodes = [str(int(node.item()) + 1) for node in elem[1:]]
                            f.write(f"{elem_id}, " + ", ".join(nodes) + "\n")
            
            # Write node sets
            if hasattr(self, 'sets_nodes'):
                for set_name, node_set in self.sets_nodes.items():
                    f.write(f"*Nset, nset={set_name}\n")
                    nodes_list = [str(node_id + 1) for node_id in sorted(node_set)]
                    # Write nodes in groups of 10 per line
                    for i in range(0, len(nodes_list), 10):
                        line_nodes = nodes_list[i:i+10]
                        f.write(", ".join(line_nodes) + "\n")
            
            # Write element sets
            if hasattr(self, 'sets_elems'):
                for set_name, elem_set in self.sets_elems.items():
                    f.write(f"*Elset, elset={set_name}\n")
                    elems_list = [str(elem_id + 1) for elem_id in sorted(elem_set)]
                    # Write elements in groups of 10 per line
                    for i in range(0, len(elems_list), 10):
                        line_elems = elems_list[i:i+10]
                        f.write(", ".join(line_elems) + "\n")
            
            # Write surfaces
            if hasattr(self, 'surfaces'):
                for surface_name, surface_data in self.surfaces.items():
                    f.write(f"*Surface, type=ELEMENT, name={surface_name}\n")
                    for elem_ids, surface_index in surface_data:
                        if isinstance(elem_ids, np.ndarray):
                            for elem_id in elem_ids:
                                f.write(f"{set_name}, S{surface_index + 1}\n")
                        else:
                            f.write(f"{set_name}, S{surface_index + 1}\n")
            
            # Write sections
            if hasattr(self, 'sections'):
                for section in self.sections:
                    set_name, material_name = section
                    f.write(f"** Section: {set_name}\n")
                    f.write(f"*Solid Section, elset={set_name}, material={material_name}\n")
            
            f.write("*End Part\n")

    class Materials():
        # materials [density, type:(0:linear, 1:neohooken), para:]

        def __init__(self) -> None:
            self.type: int
            self.mat_para: list[float]
            self.density: float = 0.0

        def read(self, origin_data: list[str], ind: int):
            while ind < len(origin_data):
                now = origin_data[ind]

                if len(now) >= 13 and now[0:13] == '*Hyperelastic':
                    self.type = 1
                    ind += 1
                    now = origin_data[ind]
                    self.mat_para = list(map(float, now.split(',')))
                    self.mat_para[0] = self.mat_para[0] * 2
                    self.mat_para[1] = 2 / (self.mat_para[1])

                if len(now) >= 8 and now[0:8] == '*Elastic':
                    self.type = 0
                    ind += 1
                    now = origin_data[ind]
                    self.mat_para = list(map(float, now.split(',')))

                if len(now) >= 8 and now[0:8] == '*Density':
                    ind += 1
                    now = origin_data[ind]
                    self.density = float(now.split(',')[0])

                if len(now) >= 9 and now[0:9] == '*Material':
                    break
                if len(now) >= 2 and now[0:2] == '**':
                    break
                ind += 1

        def write_inp(self, f: TextIOWrapper, mat_name: str):
            """
            Write material data to INP file format.
            
            Args:
                f: File object to write to
                mat_name (str): Name of the material
            """
            f.write(f"*Material, name={mat_name}\n")
            if self.density > 0:
                f.write("*Density\n")
                f.write(f"{self.density},\n")
            
            if self.type == 0:  # Linear elastic
                f.write("*Elastic\n")
                f.write(f"{self.mat_para[0]}, {self.mat_para[1]}\n")
            elif self.type == 1:  # Neo-Hookean
                f.write("*Hyperelastic, neo hooke\n")
                # Convert back from internal format
                c10 = self.mat_para[0] / 2
                d1 = 2 / self.mat_para[1]
                f.write(f"{c10}, {d1}\n")

    class Assembly():
        def __init__(self) -> None:
            self.instances: dict['str', tuple[str, np.ndarray]]
            self.sets_nodes: dict['str', set]
            self.sets_elems: dict['str', set]
            self.nodes: np.ndarray
            self.sets_elems: dict['str', set]
            self.sets_nodes: dict['str', set]

        def read(self, origin_data: list[str], ind):
            self.instances = {}
            self.sets_nodes = {}
            self.sets_elems = {}
            
            while ind < len(origin_data):
                now = origin_data[ind]
                if len(now) >= 13 and now[0:13] == '*End Assembly':
                    break
                    
                # case instance
                if len(now) >= 11 and now[0:11] == '*Instance, ':
                    # 解析实例参数
                    data_now = now.split(',')
                    instance_name = None
                    part_name = None
                    
                    for param in data_now:
                        param = param.strip()
                        if param.startswith('name='):
                            instance_name = param.split('=')[1].strip()
                        elif param.startswith('part='):
                            part_name = param.split('=')[1].strip()
                    
                    ind += 1
                    # 检查是否有变换矩阵
                    transform = np.eye(4)  # 默认单位矩阵
                    if ind < len(origin_data) and origin_data[ind][0] != '*':
                        # 解析第一行：平移变换
                        transform_data = list(map(float, origin_data[ind].split(',')))
                        if len(transform_data) >= 3:
                            # 平移变换
                            transform[0, 3] = transform_data[0]
                            transform[1, 3] = transform_data[1] 
                            transform[2, 3] = transform_data[2]
                        ind += 1
                        
                        # 检查是否有第二行：旋转变换
                        if ind < len(origin_data) and origin_data[ind][0] != '*':
                            rotation_data = list(map(float, origin_data[ind].split(',')))
                            if len(rotation_data) >= 7:
                                # 旋转轴起点
                                axis_start = np.array([rotation_data[0], rotation_data[1], rotation_data[2]])
                                # 旋转轴终点
                                axis_end = np.array([rotation_data[3], rotation_data[4], rotation_data[5]])
                                # 旋转角度（度）
                                angle_deg = rotation_data[6]
                                
                                # 计算旋转轴方向向量
                                axis_vector = axis_end - axis_start
                                axis_vector = axis_vector / np.linalg.norm(axis_vector)  # 归一化
                                
                                # 将角度转换为弧度
                                angle_rad = np.radians(angle_deg)
                                
                                # 使用Rodrigues旋转公式构建旋转矩阵
                                cos_theta = np.cos(angle_rad)
                                sin_theta = np.sin(angle_rad)
                                one_minus_cos = 1 - cos_theta
                                
                                ux, uy, uz = axis_vector
                                
                                # 构建旋转矩阵
                                rotation_matrix = np.array([
                                    [cos_theta + ux*ux*one_minus_cos, 
                                    ux*uy*one_minus_cos - uz*sin_theta, 
                                    ux*uz*one_minus_cos + uy*sin_theta],
                                    [uy*ux*one_minus_cos + uz*sin_theta, 
                                    cos_theta + uy*uy*one_minus_cos, 
                                    uy*uz*one_minus_cos - ux*sin_theta],
                                    [uz*ux*one_minus_cos - uy*sin_theta, 
                                    uz*uy*one_minus_cos + ux*sin_theta, 
                                    cos_theta + uz*uz*one_minus_cos]
                                ])
                                
                                # 将旋转矩阵嵌入到4x4变换矩阵中
                                transform[:3, :3] = rotation_matrix
                                
                                # 如果旋转轴不过原点，需要调整平移部分
                                # T = T_translate * T_rotate_about_axis
                                # 先将点移动到旋转轴起点，然后旋转，再移回
                                translation_adjustment = axis_start - rotation_matrix @ axis_start
                                transform[0, 3] += translation_adjustment[0]
                                transform[1, 3] += translation_adjustment[1]
                                transform[2, 3] += translation_adjustment[2]
                                
                            ind += 1

                    self.instances[instance_name] = (part_name, transform)
                    continue
        
                # case node set
                if len(now) >= 12 and now[0:12] == '*Nset, nset=':
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        self.sets_nodes[name] = set(
                            (np.array(datalist, dtype=int) - 1).tolist())
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        self.sets_nodes[name] = set(
                            (np.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                        ind += 1
                    continue
        
                # case element set  
                if len(now) >= 14 and now[0:14] == '*Elset, elset=':
                    data_now = now.split('=')[1].split(',')
                    for ii in range(len(data_now)):
                        data_now[ii] = data_now[ii].strip()
                    name = data_now[0].strip()
                    instance_name = None
                    
                    # 检查是否指定了实例
                    for param in data_now:
                        if param.strip().startswith('instance='):
                            instance_name = param.strip().split('=')[1]
                            break
                    
                    ind += 1
                    if not 'generate' in data_now:
                        ind0 = ind
                        now = origin_data[ind]
                        while now[0] != '*':
                            ind += 1
                            now = origin_data[ind]
                            ind1 = ind
                        datalist = [[
                            int(i) for i in row.replace('\n', '').replace(
                                ',', ' ').strip().split()
                        ] for row in origin_data[ind0:ind1]]
                        datalist = [
                            element for sublist in datalist for element in sublist
                        ]
                        if instance_name:
                            # 如果指定了实例，使用实例名作为前缀
                            full_name = f"{instance_name}.{name}"
                        else:
                            full_name = name
                        self.sets_elems[full_name] = set(
                            (np.array(datalist, dtype=int) - 1).tolist())
                    else:
                        now = list(map(int, origin_data[ind].split(',')))
                        if instance_name:
                            full_name = f"{instance_name}.{name}"
                        else:
                            full_name = name
                        self.sets_elems[full_name] = set(
                            (np.arange(now[0], now[1]+1, now[2]) - 1).tolist())
                        ind += 1
                    continue
        
                # case node
                if len(now) >= 5 and now[0:5] == '*Node':
                    ind += 1
                    ind0 = ind
                    now = origin_data[ind]
                    while now[0] != '*':
                        ind += 1
                        now = origin_data[ind]
                        ind1 = ind
                    datalist = [[
                        float(i) for i in row.replace('\n', '').strip().split(',')
                    ] for row in origin_data[ind0:ind1]]
                    self.nodes = np.array(datalist, dtype=float)
                    self.nodes[:, 0] -= 1
                    continue
        
                ind += 1

        def write_inp(self, f: TextIOWrapper):
            """
            Write assembly data to INP file format.
            
            Args:
                f: File object to write to
            """
            f.write("*Assembly, name=Assembly\n")
            f.write("**\n")
            
            # Write instances
            if hasattr(self, 'instances'):
                for instance_name, (part_name, transform) in self.instances.items():
                    f.write(f"*Instance, name={instance_name}, part={part_name}\n")
                    
                    # Write transformation if not identity
                    if not np.allclose(transform, np.eye(4)):
                        # Extract translation
                        translation = transform[:3, 3]
                        rotation_matrix = transform[:3, :3]
                        
                        # Check if there's a translation
                        has_translation = not np.allclose(translation, [0, 0, 0])
                        
                        # Check if there's a rotation (rotation matrix is not identity)
                        has_rotation = not np.allclose(rotation_matrix, np.eye(3))
                        
                        if has_translation and not has_rotation:
                            # Only translation
                            f.write(f"{translation[0]}, {translation[1]}, {translation[2]}\n")
                        
                        elif has_rotation:
                            # Extract rotation axis and angle from rotation matrix
                            # Using Rodrigues formula in reverse
                            trace = np.trace(rotation_matrix)
                            angle_rad = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                            angle_deg = np.degrees(angle_rad)
                            
                            if np.abs(angle_rad) < 1e-8:
                                # No rotation (identity matrix)
                                if has_translation:
                                    f.write(f"{translation[0]}, {translation[1]}, {translation[2]}\n")
                            elif np.abs(angle_rad - np.pi) < 1e-8:
                                # 180 degree rotation - special case
                                # Find the eigenvector corresponding to eigenvalue 1
                                eigenvals, eigenvecs = np.linalg.eig(rotation_matrix)
                                axis_idx = np.argmin(np.abs(eigenvals - 1))
                                axis = eigenvecs[:, axis_idx].real
                                axis = axis / np.linalg.norm(axis)
                                
                                # Choose a point on the rotation axis (origin for simplicity)
                                axis_start = np.array([0.0, 0.0, 0.0])
                                axis_end = axis_start + axis
                                
                                if has_translation:
                                    f.write(f"{translation[0]}, {translation[1]}, {translation[2]}\n")
                                f.write(f"{axis_start[0]}, {axis_start[1]}, {axis_start[2]}, "
                                       f"{axis_end[0]}, {axis_end[1]}, {axis_end[2]}, {angle_deg}\n")
                            else:
                                # General rotation
                                # Extract rotation axis
                                axis = np.array([
                                    rotation_matrix[2, 1] - rotation_matrix[1, 2],
                                    rotation_matrix[0, 2] - rotation_matrix[2, 0],
                                    rotation_matrix[1, 0] - rotation_matrix[0, 1]
                                ]) / (2 * np.sin(angle_rad))
                                
                                axis = axis / np.linalg.norm(axis)  # Normalize
                                
                                # For writing, we need to specify two points on the rotation axis
                                # We'll use origin as start point and axis direction as end point
                                axis_start = np.array([0.0, 0.0, 0.0])
                                axis_end = axis_start + axis
                                
                                # If there's translation, write it first
                                if has_translation:
                                    # For combined translation and rotation, we need to account for
                                    # the fact that rotation happens first, then translation
                                    # But in INP format, translation is applied to the original position
                                    # So we write the final translation
                                    f.write(f"{translation[0]}, {translation[1]}, {translation[2]}\n")
                                
                                # Write rotation: axis_start_x, axis_start_y, axis_start_z, 
                                #                 axis_end_x, axis_end_y, axis_end_z, angle_degrees
                                f.write(f"{axis_start[0]}, {axis_start[1]}, {axis_start[2]}, "
                                       f"{axis_end[0]}, {axis_end[1]}, {axis_end[2]}, {angle_deg}\n")
                    
                    f.write("*End Instance\n")
                    f.write("**\n")
            
            # Write assembly-level node sets
            if hasattr(self, 'sets_nodes'):
                for set_name, node_set in self.sets_nodes.items():
                    f.write(f"*Nset, nset={set_name}\n")
                    nodes_list = [str(node_id + 1) for node_id in sorted(node_set)]
                    for i in range(0, len(nodes_list), 10):
                        line_nodes = nodes_list[i:i+10]
                        f.write(", ".join(line_nodes) + "\n")
            
            # Write assembly-level element sets
            if hasattr(self, 'sets_elems'):
                for set_name, elem_set in self.sets_elems.items():
                    f.write(f"*Elset, elset={set_name}\n")
                    elems_list = [str(elem_id + 1) for elem_id in sorted(elem_set)]
                    for i in range(0, len(elems_list), 10):
                        line_elems = elems_list[i:i+10]
                        f.write(", ".join(line_elems) + "\n")
            
            f.write("*End Assembly\n")

    def __init__(self) -> None:
        """
        Initializes the FEA_INP class.

        This method initializes the FEA_INP class and sets up the necessary attributes.

        Args:
            None

        Returns:
            None
        """

        self.part: dict['str', FEA_INP.Parts] = {}
        self.material: dict['str', FEA_INP.Materials] = {}
        self.assemble: FEA_INP.Assembly = FEA_INP.Assembly()
        self.disp_result = []

    def read_inp(self, path):
        """
        Reads an INP file.

        This method reads an INP file and extracts the necessary information such as assembly, parts, and materials.

        Args:
            path (str): The path to the INP file.

        Returns:
            None
        """
        threads = []
        self.part = {}
        self.material = {}

        f = open(path)
        origin_data = f.readlines()
        f.close()
        for findex in range(len(origin_data)):
            now = origin_data[findex]
            if len(now) >= 16 and now[0:16] == '*Assembly, name=':
                name = now[16:].replace('\n', '').strip()
                self.assemble = FEA_INP.Assembly()
                self.assemble.read(origin_data=origin_data, ind=findex + 1)

            if len(now) >= 12 and now[0:12] == '*Part, name=':
                name = now[12:].replace('\n', '').strip()
                self.part[name] = FEA_INP.Parts()
                self.part[name].read(origin_data=origin_data, ind=findex + 1)

            if len(now) >= 16 and now[0:16] == '*Material, name=':
                name = now[16:].replace('\n', '').strip()
                self.material[name] = FEA_INP.Materials()
                self.material[name].read(
                    origin_data=origin_data,
                    ind=findex + 1
                )

        for p_key in self.part.keys():
            p = self.part[p_key]
            for sec in p.sections:
                index = np.array(list(p.sets_elems[sec[0]]), dtype=int)
                mat = self.material[sec[1]]
                p.elems_material[index, 0] = index.astype(p.elems_material.dtype)
                p.elems_material[index, 1] = mat.density
                p.elems_material[index, 2] = mat.type
                p.elems_material[index, 3] = mat.mat_para[0]
                p.elems_material[index, 4] = mat.mat_para[1]

    def write_inp(self, path: str):
        """
        Write the FEA model to an INP file.
        
        Args:
            path (str): Path where to save the INP file
        """
        with open(path, 'w') as f:
            f.write("** Job name: Model-1 Model name: Model-1\n")
            f.write("** Generated by FEA_INP class\n")
            f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
            f.write("**\n")
            
            # Write materials
            for mat_name, material in self.material.items():
                material.write_inp(f, mat_name)
            
            f.write("**\n")
            
            # Write parts
            for part_name, part in self.part.items():
                part.write_inp(f, part_name)
                f.write("**\n")
            
            # Write assembly if it exists
            if hasattr(self, 'assemble') and self.assemble is not None:
                self.assemble.write_inp(f)

