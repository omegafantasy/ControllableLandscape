from base import *


def find_data(name: str, data: list):
    for dic in data:
        if dic["name"] == name:
            return dic
    raise ValueError("No such data")


def euler2quaternion(x: float, y: float, z: float) -> Tuple[float, float, float, float]:
    x, y, z = x / 2, y / 2, z / 2
    c1, s1 = cos(x), sin(x)
    c2, s2 = cos(y), sin(y)
    c3, s3 = cos(z), sin(z)
    return (
        s1 * c2 * c3 - c1 * s2 * s3,
        c1 * s2 * c3 + s1 * c2 * s3,
        c1 * c2 * s3 - s1 * s2 * c3,
        c1 * c2 * c3 + s1 * s2 * s3,
    )


def xy2idx(x: float, y: float) -> Tuple[float, float]:
    return (x + RL) * MAP_W / ((W + 2) * RL), (y + RL) * MAP_H / ((H + 2) * RL)


def eu_dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def ang_dist(a1, a2):
    a1 = (a1 % (2 * pi) + 2 * pi) % (2 * pi)
    a2 = (a2 % (2 * pi) + 2 * pi) % (2 * pi)
    return min(abs(a1 - a2), 2 * pi - abs(a1 - a2))


def p(arg1, arg2=None):
    if arg2 is None:
        return np.array(arg1)
    return np.array([arg1, arg2])


def get_height(height_map: List[List[float]], x: float, y: float) -> float:
    # note that height map is based on an extended grid (W+2)*(H+2)
    idx_x, idx_y = xy2idx(x, y)
    x1, y1 = int(idx_x), int(idx_y)
    x2, y2 = x1 + 1, y1 + 1
    if x1 < 0 or x2 >= MAP_W or y1 < 0 or y2 >= MAP_H:
        return WATER_HEIGHT
    xx, yy = idx_x - x1, idx_y - y1
    w1, w2, w3, w4 = (1 - xx) * (1 - yy), (1 - xx) * yy, xx * (1 - yy), xx * yy
    return max(
        min(w1 * height_map[x1][y1] + w2 * height_map[x1][y2] + w3 * height_map[x2][y1] + w4 * height_map[x2][y2], 1),
        0,
    )


def output_height_map(height_map: List[List[float]], width: int, height: int, name: str, suffix: str) -> None:
    # use cv2 to convert height_map to a width*height png
    # height_map(0,0) should be the bottom left corner, width is x, height is y
    assert width >= height
    img = np.zeros((width, width, 3), np.uint8)
    for i in range(width):
        for j in range(height):
            grey = round(height_map[i][j] * 255)
            img[width - j - 1][i] = (grey, grey, grey)

    outpath = "outputs/" + name + suffix + ".png"
    cv2.imwrite(outpath, img)


def output_label_map(label_map: List[List[List[float]]], width: int, height: int, name: str, suffix: str) -> None:
    # use cv2 to convert label_map to a width*height png
    # label_map(0,0) should be the bottom left corner, width is x, height is y
    assert width >= height
    img = np.zeros((width, width, 3), np.uint8)
    for i in range(width):
        for j in range(height):
            label = label_map[i][j]
            label = [label[1] + label[2], label[3], label[4]]
            sum = label[0] + label[1] + label[2]
            if sum == 0:
                img[width - j - 1][i] = (0, 0, 0)
            else:
                if sum > 1:
                    label = [label[0] / sum, label[1] / sum, label[2] / sum]
                lb = label[0]
                lg = label[1] / (1 - label[0]) if label[0] < 1 else 0
                lr = label[2] / (1 - label[0] - label[1]) if label[0] + label[1] < 1 else 0
                img[width - j - 1][i] = (round(lb * 255), round(lg * 255), round(lr * 255))

    outpath = "outputs/" + name + suffix + ".png"
    cv2.imwrite(outpath, img)


def output_scene(
    all_obj_infos: Tuple[List[Tuple[float, float, float, float]], str],
    view_points: List[Tuple[float, float, float, float, float]],
    data: dict,
    gen_idx: int,
    suffix: str,
) -> None:
    type2infos = {}
    for obj_infos, obj_type in all_obj_infos:
        if obj_type not in type2infos:
            type2infos[obj_type] = []
        type2infos[obj_type].extend(obj_infos)

    out_tree = []
    for obj_type in type2infos:
        type_dict = {}
        type_data = None
        for key in data:
            for dic in data[key]:
                if dic["name"] == obj_type:
                    type_data = dic
                    break
        type_dict["name"] = obj_type
        type_dict["path"] = type_data["path"]
        base_x, base_y, base_z = (
            type_data["cbottom"][0],
            type_data["cbottom"][1],
            type_data["cbottom"][2],
        )

        transforms = []
        for obj_info in type2infos[obj_type]:
            # print(obj_info)
            transform = {}
            forw = obj_info[3]
            scalex, scaley, scalez = 1, 1, 1
            if obj_type == "Flower_A" or obj_type == "Flower_B" or obj_type == "Flower_C" or obj_type == "Flower_D":
                rand_scale = random.uniform(1.5, 2)
                scalex = scaley = scalez = rand_scale
            elif (
                obj_type == "Lotus_A"
                or obj_type == "Lotus_B"
                or obj_type == "Lotus_Flower_A"
                or obj_type == "Lotus_Flower_B"
            ):
                rand_scale = random.uniform(1.2, 1.8)
                scalex = scaley = scalez = rand_scale
            elif obj_type == "Bamboo_A" or obj_type == "Bamboo_B" or obj_type == "Bamboo_C":
                scaley = 1.3
            elif obj_type == "Building_A" or obj_type == "Building_B" or obj_type == "Building_C":
                scaley = 1.3
            elif obj_type == "Wall_400x300":
                scaley = 1.5
            elif obj_type == "BushBig":
                rand_scale = random.uniform(0.7, 1.5)
                scalex = scalez = rand_scale
                scaley = rand_scale * random.uniform(1.6, 2.5)
            elif obj_type == "Plant_A" or obj_type == "Plant_B" or obj_type == "shaggy_soldier":
                scalex = scaley = scalez = 1.5
            elif obj_type == "SM_SquareBush":
                scaley = 1.5
            elif obj_type == "TH_Rock_A" or obj_type == "TH_Rock_B":
                rand_scale = random.uniform(2, 3)
                scalex = scaley = scalez = rand_scale
            elif obj_type == "Rock_A" or obj_type == "Rock_B" or obj_type == "Rock_C":
                rand_scale = random.uniform(7.5, 10)
                scaley = rand_scale
                scalex = scalez = 5
            elif obj_type == "hugetree":
                rand_scale = random.uniform(1.5, 2)
                scalex = scaley = scalez = rand_scale
            elif obj_type == "Bush01" or obj_type == "Bush02" or obj_type == "Bush03":
                rand_scale = random.uniform(1.5, 2)
                scalex = scaley = scalez = rand_scale
            elif obj_type == "SM_RoundBush" or obj_type == "SM_RoundBush2":
                scalex = scaley = scalez = 1.3

            rotated_x, rotated_z = base_x * cos(-forw) - base_z * sin(-forw), base_x * sin(-forw) + base_z * cos(-forw)
            transform["position"] = {
                "x": obj_info[0] + rotated_x * scalex,
                "y": obj_info[1] - base_y,
                "z": obj_info[2] - rotated_z * scalez,
            }
            quaternion = euler2quaternion(0, -forw, 0)
            transform["rotation"] = {"x": quaternion[0], "y": quaternion[1], "z": quaternion[2], "w": quaternion[3]}
            transform["scale"] = {"x": scalex, "y": scaley, "z": scalez}
            transforms.append(transform)
        type_dict["transforms"] = transforms
        out_tree.append(type_dict)

    out = {"tree": out_tree}
    out["height_map_path"] = "height_map_" + str(gen_idx) + suffix + ".png"
    out["label_map_path"] = "label_map_" + str(gen_idx) + suffix + ".png"
    out["max_height"] = MAX_HEIGHT
    out["water_height"] = WATER_HEIGHT * MAX_HEIGHT
    out["map_width"] = MAP_W
    out["map_height"] = MAP_H
    out["real_width"] = (W + 2) * RL
    out["real_height"] = (H + 2) * RL * MAP_W / MAP_H
    out["width_offset"] = -RL
    out["height_offset"] = -RL

    out_viewpoints = []
    for i in range(len(view_points)):
        out_viewpoints.append(
            {
                "x": view_points[i][0],
                "y": view_points[i][1],
                "z": view_points[i][2],
                "xrot": view_points[i][3],
                "yrot": view_points[i][4],
            }
        )
    out["viewpoints"] = out_viewpoints

    outpath = "outputs/scene_" + str(gen_idx) + suffix + ".json"
    with open(outpath, "w") as outf:
        json.dump(out, outf)


def random_placing(
    poly_: Polygon, size_list: List[Tuple[float, float]], ratio: float, override: bool, buffer: float = 0
) -> List[Tuple[float, float, int]]:
    """
    randomly place objects in the polygon
    """
    poly = poly_.buffer(-buffer) if buffer > 0 else deepcopy(poly_)
    if poly.area<0.1:
        return []
    target_area = poly.area * ratio
    now_area = 0
    type_areas = [size[0] * size[1] for size in size_list]
    point_and_types = []
    minx, minz, maxx, maxz = poly.bounds
    num_types = len(size_list)

    continuous_fail = 0
    while True:
        typ = random.randint(0, num_types - 1)
        x = random.uniform(minx, maxx)
        z = random.uniform(minz, maxz)
        if poly.contains(Point(x, z)):
            valid = True
            if not override:
                for px, pz, p_typ in point_and_types:
                    if (
                        abs(px - x) < (size_list[p_typ][0] + size_list[typ][0]) / 2
                        and abs(pz - z) < (size_list[p_typ][1] + size_list[typ][1]) / 2
                    ):
                        valid = False
                        break
            if valid:
                point_and_types.append((x, z, typ))
                now_area += type_areas[typ]
                continuous_fail = 0
                if now_area >= target_area:
                    break
            else:
                continuous_fail += 1
                if continuous_fail > 50:
                    override = True
    return point_and_types


def grid_random_placing(
    poly_: Polygon, size_list: List[Tuple[float, float]], gx: float, gz: float
) -> List[Tuple[float, float, int]]:
    """
    gx, gz: size of the grid
    randomly place objects in each grid
    """
    poly = deepcopy(poly_)
    point_and_types = []
    minx, minz, maxx, maxz = poly.bounds
    wnum, hnum = int((maxx - minx) / gx), int((maxz - minz) / gz)
    if wnum == 0 or hnum == 0:
        return []
    gx, gz = (maxx - minx) / wnum, (maxz - minz) / hnum
    num_types = len(size_list)
    for i in range(wnum):
        for j in range(hnum):
            typ = random.randint(0, num_types - 1)
            xl, xh, zl, zh = minx + i * gx, minx + (i + 1) * gx, minz + j * gz, minz + (j + 1) * gz
            mx, mz = size_list[typ][0], size_list[typ][1]
            for k in range(10):
                x, z = random.uniform(xl + mx / 2, xh - mx / 2), random.uniform(zl + mz / 2, zh - mz / 2)
                if xh - xl <= mx:
                    x = (xh + xl) / 2
                if zh - zl <= mz:
                    z = (zh + zl) / 2
                if poly.contains(Point(x, z)):
                    point_and_types.append((x, z, typ))
                    break
    return point_and_types


def group_random_placing(
    poly_: Polygon,
    size_list: List[Tuple[float, float]],
    ratio: float,
    buffer: float,
    group_range: float,
    group_num: int,
) -> List[Tuple[float, float, int]]:
    poly = poly_.buffer(-group_range - buffer)
    target_area = poly.area * ratio
    now_area = 0
    type_areas = [size[0] * size[1] for size in size_list]
    point_and_types = []
    minx, minz, maxx, maxz = poly.bounds
    num_types = len(size_list)

    while True:
        typ = random.randint(0, num_types - 1)
        x = random.uniform(minx, maxx)
        z = random.uniform(minz, maxz)
        if poly.contains(Point(x, z)):
            for i in range(group_num):
                rand_angle = random.uniform(0, 2 * pi)
                rand_radius = random.uniform(0, group_range)
                new_x = x + rand_radius * cos(rand_angle)
                new_z = z + rand_radius * sin(rand_angle)
                point_and_types.append((new_x, new_z, typ))
                now_area += type_areas[typ]
                if now_area >= target_area:
                    break
        if now_area >= target_area:
            break

    return point_and_types


def maze_random_placing(
    poly_: Polygon, size_list: List[Tuple[float, float]], gx: float, gz: float, ratio: float
) -> List[Tuple[float, float, int]]:
    """
    gx, gz: size of the grid
    randomly generate a maze and place objects (not exceeding ratio)
    """
    poly = deepcopy(poly_)
    point_and_types = []
    minx, minz, maxx, maxz = poly.bounds
    wnum, hnum = int((maxx - minx) / gx), int((maxz - minz) / gz)
    if wnum == 0 or hnum == 0:
        return []
    gx, gz = (maxx - minx) / wnum, (maxz - minz) / hnum
    target_area = poly.area * ratio
    target_num = int(target_area / (gx * gz)) + 1
    num_types = len(size_list)

    road = set()
    dxy = [0, 2, 0, -2, 0]

    def maze_dfs(curr_pos):
        road.add(curr_pos)
        stack = [curr_pos]
        while stack:
            curr_pos = stack.pop()
            p = [0, 1, 2, 3]
            random.shuffle(p)
            for i in p:
                next_pos = (curr_pos[0] + dxy[i], curr_pos[1] + dxy[i + 1])
                if 0 <= next_pos[0] < wnum and 0 <= next_pos[1] < hnum and next_pos not in road:
                    road.add(((curr_pos[0] + next_pos[0]) // 2, (curr_pos[1] + next_pos[1]) // 2))
                    road.add(next_pos)
                    stack.append(next_pos)

    maze_dfs((0, 0))

    candidates = []
    for i in range(wnum):
        for j in range(hnum):
            x, z = (i + 0.5) * gx + minx, (j + 0.5) * gz + minz
            if poly.contains(Point(x, z)) and ((i, j) not in road):
                candidates.append((x, z))
    random.shuffle(candidates)
    for i in range(min(target_num, len(candidates))):
        typ = random.randint(0, num_types - 1)
        point_and_types.append((candidates[i][0], candidates[i][1], typ))
    return point_and_types


def along_bordered_placing(poly_: Polygon, mx: float, mz: float, step: float, layers: int) -> List[Tuple[float, float]]:
    """
    mx, mz: boundbox of the object
    step: distance between two objects
    layers: number of layers
    place objects along the border of the polygon
    """
    poly = deepcopy(poly_)
    points = []
    for k in range(layers):
        if k == 0:
            poly = poly.buffer(-max(mx, mz) / 2)
        else:
            poly = poly.buffer(-step)
        if not isinstance(poly, Polygon):
            break
        boundary_ring = poly.exterior
        layer_num = max(int(boundary_ring.length / step) - 1, 1)
        for i in range(layer_num):
            point = boundary_ring.interpolate(i * step)
            points.append((point.x, point.y))
    return points


def override_height(
    height_map: List[List[float]], poly: Polygon, height: float, buffer: float
) -> List[Tuple[int, int]]:
    """
    modify values in height_map
    region in poly will be set to height
    units near poly will be smoothed, buffer is the distance
    """
    points = [xy2idx(x, y) for x, y in poly.exterior.coords]
    idx_poly = Polygon(points)
    # decide what units are covered by the idx_poly
    minx, miny, maxx, maxy = idx_poly.bounds
    ratio = max(MAP_W, MAP_H) / (RL * max(W, H))
    x1, z1 = int(minx - buffer * ratio), int(miny - buffer * ratio)
    x2, z2 = int(maxx + buffer * ratio) + 1, int(maxy + buffer * ratio) + 1
    idxs = []
    for i in range(x1, x2):
        for j in range(z1, z2):
            if i < 0 or i >= MAP_W or j < 0 or j >= MAP_H:
                continue
            pcenter = Point(i + 0.5, j + 0.5)
            if idx_poly.contains(pcenter):
                height_map[i][j] = height
                idxs.append((i, j))
            else:
                dis = pcenter.distance(idx_poly) / ratio
                if dis < buffer:
                    height_map[i][j] = height * (1 - dis / buffer) + height_map[i][j] * dis / buffer

    return idxs


def along_line_placing(
    point_locs: List[Tuple[float, float]], step: float, width: float
) -> List[Tuple[float, float, float]]:
    loc_and_forws = []
    last_idx, last_loc, last_forw = 0, p(deepcopy(point_locs[0])), None
    while True:
        next_idx = None
        for i in range(last_idx, len(point_locs)):
            next_idx = i
            if eu_dist(last_loc, point_locs[i]) > step:
                break
        if eu_dist(last_loc, point_locs[next_idx]) <= 0.25:
            break

        forw_vec = (point_locs[next_idx][0] - last_loc[0], point_locs[next_idx][1] - last_loc[1])
        forw = np.arctan2(forw_vec[1], forw_vec[0])
        sp, ep = None, None
        if last_forw is None:
            sp, ep = last_loc, last_loc + step * p(cos(forw), sin(forw))
        else:
            a_dist = ang_dist(forw, last_forw)
            if eu_dist(last_loc, point_locs[next_idx]) + sin(a_dist / 2) * width * 0.75 < step:
                ep = point_locs[next_idx]
                sp = ep - step * p(cos(forw), sin(forw))
            else:
                sp = last_loc - sin(a_dist / 2) * width * p(cos(forw), sin(forw)) * 0.75
                ep = sp + step * p(cos(forw), sin(forw))

        midx, midy = (sp[0] + ep[0]) / 2, (sp[1] + ep[1]) / 2
        loc_and_forws.append((midx, midy, forw))

        xl, yl, xh, yh = (
            min(sp[0], ep[0]) - 1e-3,
            min(sp[1], ep[1]) - 1e-3,
            max(sp[0], ep[0]) + 1e-3,
            max(sp[1], ep[1]) + 1e-3,
        )
        xlast, ylast = point_locs[-1][0], point_locs[-1][1]
        if xl <= xlast <= xh and yl <= ylast <= yh:
            break
        last_loc, last_forw, last_idx = ep, forw, next_idx

    return loc_and_forws


def build_zigzagbridge(
    height_map: List[List[float]], point_locs: List[Tuple[float, float]], data: dict
) -> List[Tuple[float, float, float, float]]:
    step, height, width = data["size"][0], data["size"][1], data["size"][2]
    loc_and_forws = along_line_placing(point_locs, step, width)
    start_obj, end_obj = loc_and_forws[0], loc_and_forws[-1]
    sp = p(start_obj[0], start_obj[1]) - step / 2 * p(cos(start_obj[2]), sin(start_obj[2]))
    ep = p(end_obj[0], end_obj[1]) + step / 2 * p(cos(end_obj[2]), sin(end_obj[2]))
    lakeside_h = max(min(get_height(height_map, sp[0], sp[1]), get_height(height_map, ep[0], ep[1])), 0.1)
    base_height = lakeside_h * MAX_HEIGHT - (height - 1)

    obj_infos = []
    for loc_and_forw in loc_and_forws:
        obj_infos.append((loc_and_forw[0], base_height, loc_and_forw[1], loc_and_forw[2]))
    return obj_infos


def build_bridge(
    height_map: List[List[float]], point_locs: List[Tuple[float, float]], data: dict
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
    length = data["size"][0]
    edge_length = 0
    for i in range(len(point_locs) - 1):
        edge_length += eu_dist(point_locs[i], point_locs[i + 1])
    mid_idx, now_length = -1, 0
    for i in range(len(point_locs) - 1):
        if now_length / edge_length >= 0.5:
            mid_idx = i
            break
        now_length += eu_dist(point_locs[i], point_locs[i + 1])
    idx1, idx2 = deepcopy(mid_idx), deepcopy(mid_idx)
    while True:
        if idx1 > 0:
            idx1 -= 1
        if eu_dist(point_locs[idx1], point_locs[idx2]) >= length:
            break
        if idx2 < len(point_locs) - 1:
            idx2 += 1
        if eu_dist(point_locs[idx1], point_locs[idx2]) >= length:
            break
    sp, ep = p(point_locs[idx1]), p(point_locs[idx2])
    midp, forw = (sp + ep) / 2, np.arctan2(ep[1] - sp[1], ep[0] - sp[0])

    sp, ep = midp - length / 2 * p(cos(forw), sin(forw)), midp + length / 2 * p(cos(forw), sin(forw))
    poly1, poly2 = None, None
    edges1 = [(point_locs[i], point_locs[i + 1]) for i in range(idx1)] + [(point_locs[idx1], sp)]
    edges2 = [(ep, point_locs[idx2])] + [(point_locs[i], point_locs[i + 1]) for i in range(idx2, len(point_locs) - 1)]
    for edge in edges1:
        p1, p2 = p(edge[0]), p(edge[1])
        tang = p(p2[0] - p1[0], p2[1] - p1[1])
        tang = tang / np.linalg.norm(tang)
        norm = p(p2[1] - p1[1], p1[0] - p2[0])
        norm = norm / np.linalg.norm(norm)
        poly = Polygon(
            [
                p1 + norm * MAIN_ROAD_WIDTH / 2 - tang,
                p1 - norm * MAIN_ROAD_WIDTH / 2 - tang,
                p2 - norm * MAIN_ROAD_WIDTH / 2 + tang,
                p2 + norm * MAIN_ROAD_WIDTH / 2 + tang,
            ]
        )
        if poly1 is None:
            poly1 = poly
        else:
            poly1 = poly1.union(poly)
    for edge in edges2:
        p1, p2 = p(edge[0]), p(edge[1])
        tang = p(p2[0] - p1[0], p2[1] - p1[1])
        tang = tang / np.linalg.norm(tang)
        norm = p(p2[1] - p1[1], p1[0] - p2[0])
        norm = norm / np.linalg.norm(norm)
        poly = Polygon(
            [
                p1 + norm * MAIN_ROAD_WIDTH / 2 - tang,
                p1 - norm * MAIN_ROAD_WIDTH / 2 - tang,
                p2 - norm * MAIN_ROAD_WIDTH / 2 + tang,
                p2 + norm * MAIN_ROAD_WIDTH / 2 + tang,
            ]
        )
        if poly2 is None:
            poly2 = poly
        else:
            poly2 = poly2.union(poly)
    poly1, poly2 = poly1.simplify(0, False), poly2.simplify(0, False)

    base_height_ratio = 0.1
    road_idxs = override_height(height_map, poly1, base_height_ratio, 3) + override_height(
        height_map, poly2, base_height_ratio, 3
    )
    obj_infos = [(midp[0], base_height_ratio * MAX_HEIGHT - 1.4, midp[1], forw)]
    return obj_infos, road_idxs


def build_wall(
    height_map: List[List[float]], point_locs: List[Tuple[float, float]], data: dict
) -> List[Tuple[float, float, float, float]]:
    step, width = 4, data["size"][2]
    locs_and_forws = along_line_placing(point_locs, step, width)

    obj_infos = []
    for loc_and_forw in locs_and_forws:
        midp, forw = p(loc_and_forw[0], loc_and_forw[1]), loc_and_forw[2]
        sp, ep = midp - step / 2 * p(cos(forw), sin(forw)), midp + step / 2 * p(cos(forw), sin(forw))
        h1, h2 = get_height(height_map, sp[0], sp[1]), get_height(height_map, ep[0], ep[1])
        obj_infos.append((midp[0], min(h1, h2) * MAX_HEIGHT, midp[1], forw))
    return obj_infos


def build_entrance(
    height_map: List[List[float]],
    point_locs: List[Tuple[float, float]],
    data: dict,
    entrance_points: List[Tuple[float, float]],
) -> List[Tuple[float, float, float, float]]:
    step, width = 4, data["size"][2]
    locs_and_forws = along_line_placing(point_locs, step, width)

    obj_infos = []
    for loc_and_forw in locs_and_forws:
        midp, forw = p(loc_and_forw[0], loc_and_forw[1]), loc_and_forw[2]
        far_from_entrance = True
        for entrance_point in entrance_points:
            if eu_dist(midp, entrance_point) < 5:
                far_from_entrance = False
                break
        if far_from_entrance:
            sp, ep = midp - step / 2 * p(cos(forw), sin(forw)), midp + step / 2 * p(cos(forw), sin(forw))
            h1, h2 = get_height(height_map, sp[0], sp[1]), get_height(height_map, ep[0], ep[1])
            obj_infos.append((midp[0], min(h1, h2) * MAX_HEIGHT, midp[1], forw))
    return obj_infos


def add_lotus(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    if poly.area > 4000:
        return []
    point_and_types = random_placing(poly, size_list, 0.12, True)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    model_height = WATER_HEIGHT * MAX_HEIGHT - 0.05
    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        if get_height(height_map, x, z) < WATER_HEIGHT - 0.005:
            obj_infos_dict[typ].append((x, model_height, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_lakerock(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    point_and_types = random_placing(poly, size_list, 0.001, True)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    base_height = WATER_HEIGHT * MAX_HEIGHT - 0.6
    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        if get_height(height_map, x, z) < WATER_HEIGHT - 0.005:
            obj_infos_dict[typ].append((x, base_height + random.uniform(-0.2, 0.2), z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_rockmaze(
    height_map: List[List[float]],
    poly: Polygon,
    hill_rock_data: List[dict],
    lake_rock_data: List[dict],
    tree_data: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    hillrock_size_list = [(dat["size"][0], dat["size"][2]) for dat in hill_rock_data]
    # lakerock_size_list = [(dat["size"][0], dat["size"][2]) for dat in lake_rock_data]
    tree_size_list = [(dat["size"][0], dat["size"][2]) for dat in tree_data]
    # lakerock_point_and_types = maze_random_placing(poly, lakerock_size_list, 3, 3, 0.02)
    hillrock_point_and_types = group_random_placing(poly, hillrock_size_list, 0.004, 2, 2, 3)
    tree_point_and_types = random_placing(poly, tree_size_list, 0.2, True)
    hillrock_obj_infos_dict, lakerock_obj_infos_dict, tree_obj_infos_dict = {}, {}, {}
    for typ_idx in range(len(hill_rock_data)):
        hillrock_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(lake_rock_data)):
        lakerock_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(tree_data)):
        tree_obj_infos_dict[typ_idx] = []

    for point_and_type in hillrock_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        hillrock_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    # for point_and_type in lakerock_point_and_types:
    #     x, z, typ = point_and_type
    #     x += random.uniform(-1, 1)
    #     z += random.uniform(-1, 1)
    #     y = get_height(height_map, x, z) * MAX_HEIGHT
    #     lakerock_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in tree_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        tree_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    obj_infos_list = []
    for typ_idx in hillrock_obj_infos_dict:
        obj_infos_list.append((hillrock_obj_infos_dict[typ_idx], hill_rock_data[typ_idx]["name"]))
    for typ_idx in lakerock_obj_infos_dict:
        obj_infos_list.append((lakerock_obj_infos_dict[typ_idx], lake_rock_data[typ_idx]["name"]))
    for typ_idx in tree_obj_infos_dict:
        obj_infos_list.append((tree_obj_infos_dict[typ_idx], tree_data[typ_idx]["name"]))
    return obj_infos_list


def add_bushes(
    height_map: List[List[float]], poly: Polygon, rectbush_datas: List[dict], bigbush_datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    rectbush_size_list = [(dat["size"][0], dat["size"][2]) for dat in rectbush_datas]
    bigbush_size_list = [(dat["size"][0], dat["size"][2]) for dat in bigbush_datas]
    poly_area = poly.area
    point_and_types = None
    typeflag = 0
    if poly_area < 500:
        point_and_types = grid_random_placing(poly, rectbush_size_list, 2, 2)
    else:
        if poly_area < 2000 and random.random() < 0.7:
            point_and_types = maze_random_placing(poly, rectbush_size_list, 2, 2, 0.45)
        else:
            typeflag = 1
            point_and_types = random_placing(poly, bigbush_size_list, 0.3, True, 3)
    obj_infos_dict = {}
    length = len(rectbush_datas) if typeflag == 0 else len(bigbush_datas)
    for typ_idx in range(length):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        if typeflag == 0:
            obj_infos_list.append((obj_infos_dict[typ_idx], rectbush_datas[typ_idx]["name"]))
        else:
            obj_infos_list.append((obj_infos_dict[typ_idx], bigbush_datas[typ_idx]["name"]))
    return obj_infos_list


def add_few_trees(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    point_and_types = random_placing(poly, size_list, 0.2, True)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT - 0.2
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_bamboos(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    poly_area = poly.area
    point_and_types = None
    if poly_area < 1000:
        point_and_types = grid_random_placing(poly, size_list, 2, 2)
    else:
        point_and_types = random_placing(poly, size_list, 0.6, True)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_trees(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    poly_area = poly.area
    point_and_types = random_placing(poly, size_list, 1.0, True)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT - 0.2
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_pavilion(
    terrain_labels: List[List[List[float]]],
    height_map: List[List[float]],
    poly: Polygon,
    pav_datas: List[dict],
    tree_datas: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:  # TODO
    poly_area = poly.area
    obj_infos_list = []
    existing_polys = []
    minx, minz, maxx, maxz = poly.bounds

    # add pavilions
    bminx, bminz, bmaxx, bmaxz = minx + 4, minz + 4, maxx - 4, maxz - 4
    pav_size_list = [(dat["size"][0], dat["size"][2]) for dat in pav_datas]
    num_pavs = min(3, int(poly_area / 300))
    for k in range(num_pavs):
        x, y, z, pav_poly, pav_type = [None for _ in range(5)]
        for att in range(100):
            rand_type = random.randint(0, len(pav_size_list) - 1)
            cx, cz = random.uniform(bminx, bmaxx), random.uniform(bminz, bmaxz)
            xl, xh, zl, zh = (
                cx - pav_size_list[rand_type][0] / 2,
                cx + pav_size_list[rand_type][0] / 2,
                cz - pav_size_list[rand_type][1] / 2,
                cz + pav_size_list[rand_type][1] / 2,
            )
            building_poly = Polygon([(xl, zl), (xl, zh), (xh, zh), (xh, zl)])
            if poly.contains(building_poly):
                valid = True
                for ex_poly in existing_polys:
                    if building_poly.distance(ex_poly) < 2:
                        valid = False
                        break
                if valid:
                    heights = [
                        get_height(height_map, xl, zl),
                        get_height(height_map, xl, zh),
                        get_height(height_map, xh, zl),
                        get_height(height_map, xh, zh),
                    ]
                    if max(heights) - min(heights) > 0.01:
                        continue
                    x, y, z, pav_poly, pav_type = cx, sum(heights) / 4, cz, building_poly, rand_type
                    break
        if x is None:
            continue
        labeling_area(terrain_labels, height_map, pav_poly, 4, 0.2)
        existing_polys.append(pav_poly)
        y *= MAX_HEIGHT
        if pav_datas[pav_type]["name"] == "Pavillion_Medium":
            y -= 1
        obj_infos_list.append(([(x, y, z, random.uniform(0, 2 * pi))], pav_datas[pav_type]["name"]))

    # add trees
    tree_area = deepcopy(poly)
    for ex_poly in existing_polys:
        tree_area = tree_area.difference(ex_poly)
    tree_size_list = [(dat["size"][0], dat["size"][2]) for dat in tree_datas]
    point_and_types = random_placing(tree_area, tree_size_list, 1, True)
    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT - 0.2
        obj_infos_list.append(([(x, y, z, random.uniform(0, 2 * pi))], tree_datas[typ]["name"]))

    return obj_infos_list


def add_hugetree(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    poly_area = poly.area
    point_and_types = random_placing(poly, size_list, 0.1, True, 5)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT - 0.2
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_plantbeds(
    terrain_labels: List[List[List[float]]],
    height_map: List[List[float]],
    poly: Polygon,
    flower_datas: List[dict],
    bush_datas: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    flower_size_list, bush_size_list = [(dat["size"][0], dat["size"][2]) for dat in flower_datas], [
        (dat["size"][0], dat["size"][2]) for dat in bush_datas
    ]
    num_flower_types, num_bush_types = len(flower_size_list), len(bush_size_list)
    rx, rz, gx, gz = 7.5, 6, 9.5, 7.5
    minx, miny, maxx, maxy = poly.bounds
    wnum, hnum = int((maxx - minx) / gx), int((maxy - miny) / gz)

    is_flower = [[0 for j in range(hnum)] for i in range(wnum)]
    pattern = random.randint(0, 15)
    for i in range(wnum):
        for j in range(hnum):
            if (
                (i % 2 == 0 and j % 2 == 0 and pattern == 0)
                or (i % 2 == 0 and j % 2 == 1 and pattern == 1)
                or (i % 2 == 1 and j % 2 == 0 and pattern == 2)
                or (i % 2 == 1 and j % 2 == 1 and pattern == 3)
            ):
                is_flower[i][j] = 1
            elif (
                (i % 2 == 0 and pattern == 4)
                or (i % 2 == 1 and pattern == 5)
                or (j % 2 == 0 and pattern == 6)
                or (j % 2 == 1 and pattern == 7)
            ):
                is_flower[i][j] = 1
            elif (
                (i % 4 < 2 and j % 4 < 2 and pattern == 8)
                or (i % 4 < 2 and j % 4 >= 2 and pattern == 9)
                or (i % 4 >= 2 and j % 4 < 2 and pattern == 10)
                or (i % 4 >= 2 and j % 4 >= 2 and pattern == 11)
            ):
                is_flower[i][j] = 1
            elif pattern >= 12 and random.random() < 0.5:
                is_flower[i][j] = 1

    obj_infos_list = []
    for i in range(wnum):
        for j in range(hnum):
            cx, cz = minx + (i + 0.5) * gx, miny + (j + 0.5) * gz
            xl, xh, zl, zh = cx - rx / 2, cx + rx / 2, cz - rz / 2, cz + rz / 2
            bed_poly = Polygon([(xl, zl), (xl, zh), (xh, zh), (xh, zl)])
            if poly.contains(bed_poly):
                labeling_area(terrain_labels, height_map, bed_poly, 4, 0.2)
                if is_flower[i][j] == 0:
                    typ = random.randint(0, num_bush_types - 1)
                    point_and_types = grid_random_placing(bed_poly, [bush_size_list[typ]], 2, 2)
                    obj_infos = []
                    for point_and_type in point_and_types:
                        x, z, _ = point_and_type
                        y = get_height(height_map, x, z) * MAX_HEIGHT
                        obj_infos.append((x, y, z, random.uniform(0, 2 * pi)))
                    obj_infos_list.append((obj_infos, bush_datas[typ]["name"]))
                else:
                    typ = random.randint(0, num_flower_types - 1)
                    point_and_types = random_placing(bed_poly, [flower_size_list[typ]], 0.3, False)
                    obj_infos = []
                    for point_and_type in point_and_types:
                        x, z, _ = point_and_type
                        y = get_height(height_map, x, z) * MAX_HEIGHT - 0.05
                        obj_infos.append((x, y, z, random.uniform(0, 2 * pi)))
                    obj_infos_list.append((obj_infos, flower_datas[typ]["name"]))

    return obj_infos_list


def add_treelines(
    terrain_labels: List[List[List[float]]], height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    num_types = len(size_list)
    rx, rz, gx, gz = 2, 2, 6, 6
    minx, miny, maxx, maxy = poly.bounds
    wnum, hnum = int((maxx - minx) / gx), int((maxy - miny) / gz)
    obj_infos_list = []
    for i in range(wnum):
        for j in range(hnum):
            cx, cz = minx + (i + 0.5) * gx, miny + (j + 0.5) * gz
            xl, xh, zl, zh = cx - rx / 2, cx + rx / 2, cz - rz / 2, cz + rz / 2
            bed_poly = Polygon([(xl, zl), (xl, zh), (xh, zh), (xh, zl)])
            if poly.contains(bed_poly):
                labeling_area(terrain_labels, height_map, bed_poly, 4, 0.2)
                typ = random.randint(0, num_types - 1)
                y = get_height(height_map, cx, cz) * MAX_HEIGHT - 0.2
                obj_infos_list.append(([(cx, y, cz, random.uniform(0, 2 * pi))], datas[typ]["name"]))

    return obj_infos_list


def add_building(
    height_map: List[List[float]],
    poly: Polygon,
    buid_datas: List[dict],
    statue_datas: List[dict],
    tree_datas: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:  # TODO
    poly_area = poly.area
    obj_infos_list = []
    existing_polys = []
    minx, minz, maxx, maxz = poly.bounds
    # add main building (Building_A)
    num_main_building = 1
    if poly_area > 3000:
        num_main_building = 3
    elif poly_area > 1200:
        num_main_building = 2
    main_building_data, stata_data, statb_data = (
        find_data("Building_A", buid_datas),
        find_data("Statue_A", statue_datas),
        find_data("Statue_B", statue_datas),
    )
    rx, rz = 17, 16
    bminx, bminz, bmaxx, bmaxz = minx + rz / 2, minz + rz / 2, maxx - rz / 2, maxz - rz / 2
    if bminx > bmaxx or bminz > bmaxz:
        num_main_building = 0
    for k in range(num_main_building):
        x, y, z, f_idx, buid_poly = [None for _ in range(5)]
        for att in range(100):
            cx, cz = random.uniform(bminx, bmaxx), random.uniform(bminz, bmaxz)
            forw_idx = random.randint(0, 3)  # 0: 0, 1: pi/2, 2: pi, 3: 3pi/2
            xl, xh, zl, zh = None, None, None, None
            if forw_idx == 0 or forw_idx == 2:
                xl, xh, zl, zh = cx - rx / 2, cx + rx / 2, cz - rz / 2, cz + rz / 2
            else:
                xl, xh, zl, zh = cx - rz / 2, cx + rz / 2, cz - rx / 2, cz + rx / 2
            building_poly = Polygon([(xl, zl), (xl, zh), (xh, zh), (xh, zl)])
            if poly.contains(building_poly):
                valid = True
                for existing_poly in existing_polys:
                    if building_poly.intersects(existing_poly):
                        valid = False
                        break
                if (
                    valid
                ):  # from the front of the building, cast a ray to intersect with the polygon, calculate the distance
                    point, vec = None, None
                    if forw_idx == 0:
                        point, vec = p(cx, zl), p(0, -1)
                    elif forw_idx == 1:
                        point, vec = p(xh, cz), p(1, 0)
                    elif forw_idx == 2:
                        point, vec = p(cx, zh), p(0, 1)
                    else:
                        point, vec = p(xl, cz), p(-1, 0)
                    vec *= 1000
                    ray = LineString([point, point + vec])
                    inter = ray.intersection(poly.exterior)
                    if isinstance(inter, Point):
                        dis = eu_dist(point, (inter.x, inter.y))
                        if dis < 3:
                            heights = [
                                get_height(height_map, xl, zl),
                                get_height(height_map, xl, zh),
                                get_height(height_map, xh, zl),
                                get_height(height_map, xh, zh),
                            ]
                            if max(heights) - min(heights) > 0.02:
                                continue
                            x, y, z, f_idx, buid_poly = cx, sum(heights) / 4, cz, forw_idx, building_poly
                            break
        if x is None:
            continue
        forw = f_idx * pi / 2
        building_x, building_z = x + cos(forw + pi / 2) * 2.5, z + sin(forw + pi / 2) * 2.5
        sa_x, sa_z = x + cos(forw - pi / 2) * 5.5 - cos(forw) * 7, z + sin(forw - pi / 2) * 5.5 - sin(forw) * 7
        sb_x, sb_z = x + cos(forw - pi / 2) * 5.5 + cos(forw) * 7, z + sin(forw - pi / 2) * 5.5 + sin(forw) * 7
        obj_infos_list.append(
            (
                [(building_x, y * MAX_HEIGHT, building_z, forw)],
                main_building_data["name"],
            )
        )
        obj_infos_list.append(
            (
                [(sa_x, get_height(height_map, sa_x, sa_z) * MAX_HEIGHT, sa_z, forw + pi)],
                stata_data["name"],
            )
        )
        obj_infos_list.append(
            (
                [(sb_x, get_height(height_map, sb_x, sb_z) * MAX_HEIGHT, sb_z, forw + pi)],
                statb_data["name"],
            )
        )
        existing_polys.append(buid_poly)

    # add sub buildings (Building_B and Building_C)
    sub1_data, sub2_data = find_data("Building_B", buid_datas), find_data("Building_C", buid_datas)
    num_sub_buildings = min(8, int(poly_area / 250))
    bminx, bminz, bmaxx, bmaxz = minx + 3, minz + 3, maxx - 3, maxz - 3
    for k in range(num_sub_buildings):
        x, y, z, f_idx, buid_poly, typ = [None for _ in range(6)]
        for att in range(100):
            rand_typ = random.randint(0, 1)
            if rand_typ == 0:
                rx, rz = sub1_data["size"][0], sub1_data["size"][2]
            else:
                rx, rz = sub2_data["size"][0], sub2_data["size"][2]
            cx, cz = random.uniform(bminx, bmaxx), random.uniform(bminz, bmaxz)
            forw_idx = random.randint(0, 3)
            xl, xh, zl, zh = None, None, None, None
            if forw_idx == 0 or forw_idx == 2:
                xl, xh, zl, zh = cx - rx / 2, cx + rx / 2, cz - rz / 2, cz + rz / 2
            else:
                xl, xh, zl, zh = cx - rz / 2, cx + rz / 2, cz - rx / 2, cz + rx / 2
            building_poly = Polygon([(xl, zl), (xl, zh), (xh, zh), (xh, zl)])
            if poly.contains(building_poly):
                valid = True
                for existing_poly in existing_polys:
                    if building_poly.distance(existing_poly) < 2:
                        valid = False
                        break
                if valid:
                    point, vec = None, None
                    if forw_idx == 0:
                        point, vec = p(cx, zl), p(0, -1)
                    elif forw_idx == 1:
                        point, vec = p(xh, cz), p(1, 0)
                    elif forw_idx == 2:
                        point, vec = p(cx, zh), p(0, 1)
                    else:
                        point, vec = p(xl, cz), p(-1, 0)
                    vec *= 1000
                    ray = LineString([point, point + vec])
                    inter = ray.intersection(poly.exterior)
                    if isinstance(inter, Point):
                        dis = eu_dist(point, (inter.x, inter.y))
                        if dis < 5:
                            heights = [
                                get_height(height_map, xl, zl),
                                get_height(height_map, xl, zh),
                                get_height(height_map, xh, zl),
                                get_height(height_map, xh, zh),
                            ]
                            if max(heights) - min(heights) > 0.01:
                                continue
                            x, y, z, f_idx, buid_poly, typ = cx, sum(heights) / 4, cz, forw_idx, building_poly, rand_typ
                            break
        if x is None:
            if k == num_sub_buildings - 1 and len(existing_polys) == 0:
                k -= 1
            continue
        forw = f_idx * pi / 2
        obj_infos_list.append(
            (
                [(x, y * MAX_HEIGHT, z, forw)],
                sub1_data["name"] if typ == 0 else sub2_data["name"],
            )
        )
        existing_polys.append(buid_poly)

    # add trees
    tree_area = poly.buffer(-22)
    if tree_area.area > 0.1:
        tree_area = tree_area.simplify(0, False)
        tree_size_list = [(dat["size"][0], dat["size"][2]) for dat in tree_datas]
        point_and_types = random_placing(tree_area, tree_size_list, 0.6, True)
        obj_infos_dict = {}
        for typ_idx in range(len(tree_datas)):
            obj_infos_dict[typ_idx] = []

        for point_and_type in point_and_types:
            x, z, typ = point_and_type
            y = get_height(height_map, x, z) * MAX_HEIGHT - 0.2
            obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

        for typ_idx in obj_infos_dict:
            obj_infos_list.append((obj_infos_dict[typ_idx], tree_datas[typ_idx]["name"]))

    return obj_infos_list


def add_hillrock(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    point_and_types = group_random_placing(poly, size_list, 0.01, 2, 2, 3)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT - 0.1
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_plants(
    height_map: List[List[float]], poly: Polygon, plant_datas: List[dict], bush_datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    plant_size_list, bush_size_list = [(dat["size"][0], dat["size"][2]) for dat in plant_datas], [
        (dat["size"][0], dat["size"][2]) for dat in bush_datas
    ]
    plant_point_and_types = random_placing(poly, plant_size_list, 0.3, True)
    bush_point_and_types = random_placing(poly, bush_size_list, 0.1, True)
    plant_obj_infos_dict, bush_obj_infos_dict = {}, {}
    for typ_idx in range(len(plant_datas)):
        plant_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(bush_datas)):
        bush_obj_infos_dict[typ_idx] = []

    for point_and_type in plant_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        plant_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in bush_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        bush_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in plant_obj_infos_dict:
        obj_infos_list.append((plant_obj_infos_dict[typ_idx], plant_datas[typ_idx]["name"]))
    for typ_idx in bush_obj_infos_dict:
        obj_infos_list.append((bush_obj_infos_dict[typ_idx], bush_datas[typ_idx]["name"]))
    return obj_infos_list


def labeling_area(
    terrain_labels: List[List[List[float]]], height_map: List[List[float]], poly: Polygon, label: int, buffer: float
) -> None:
    points = [xy2idx(x, y) for x, y in poly.exterior.coords]
    idx_poly = Polygon(points)
    # decide what units are covered by the idx_poly
    minx, miny, maxx, maxy = idx_poly.bounds
    ratio = max(MAP_W, MAP_H) / (RL * max(W, H))
    x1, z1 = int(minx - buffer * ratio), int(miny - buffer * ratio)
    x2, z2 = int(maxx + buffer * ratio) + 1, int(maxy + buffer * ratio) + 1
    for i in range(x1, x2):
        for j in range(z1, z2):
            if i < 0 or i >= MAP_W or j < 0 or j >= MAP_H:
                continue
            pcenter = Point(i + 0.5, j + 0.5)
            if idx_poly.contains(pcenter):
                terrain_labels[i][j][label] += 1
                if label == 1:
                    height_map[i][j] += 0.15 / MAX_HEIGHT
                elif label == 2:
                    height_map[i][j] += 0.1 / MAX_HEIGHT
            else:
                dis = pcenter.distance(idx_poly) / ratio
                if dis < buffer:
                    rat = 1 - dis / buffer
                    terrain_labels[i][j][label] += rat
                    if label == 1:
                        height_map[i][j] += 0.15 / MAX_HEIGHT * rat
                    elif label == 2:
                        height_map[i][j] += 0.1 / MAX_HEIGHT * rat


def decide_view_points(
    height_map: List[List[float]], keyunits: List[Tuple[int, int]]
) -> List[Tuple[float, float, float, float, float]]:
    view_points = []
    # based on keyunits
    # for keyunit in keyunits:
    #     x_idx, z_idx = keyunit
    #     xl, xh, zl, zh = x_idx * RL, (x_idx + 1) * RL, z_idx * RL, (z_idx + 1) * RL
    #     for i in range(4):
    #         x, z = random.uniform(xl, xh), random.uniform(zl, zh)
    #         y = get_height(height_map, x, z) * MAX_HEIGHT + 10
    #         yrot = i * pi / 2 + pi / 4
    #         xrot = 20 * pi / 180
    #         view_points.append((x, y, z, xrot, yrot))

    # specified
    x1, x2, x3, x4, x5 = 0.05 * W * RL, 0.15 * W * RL, 0.5 * W * RL, 0.85 * W * RL, 0.95 * W * RL
    z1, z2, z3, z4, z5 = 0.05 * H * RL, 0.15 * H * RL, 0.5 * H * RL, 0.85 * H * RL, 0.95 * H * RL
    # four corners
    xrot = 25 * pi / 180
    view_points.append((x1, 0, z1, xrot, pi / 4))
    view_points.append((x1, 0, z5, xrot, -pi / 4))
    view_points.append((x5, 0, z1, xrot, 3 * pi / 4))
    view_points.append((x5, 0, z5, xrot, -3 * pi / 4))
    # four edges
    view_points.append((x2, 0, z3, xrot, pi / 4))
    view_points.append((x2, 0, z3, xrot, -pi / 4))
    view_points.append((x3, 0, z2, xrot, 3 * pi / 4))
    view_points.append((x3, 0, z2, xrot, pi / 4))
    view_points.append((x4, 0, z3, xrot, -3 * pi / 4))
    view_points.append((x4, 0, z3, xrot, 3 * pi / 4))
    view_points.append((x3, 0, z4, xrot, -pi / 4))
    view_points.append((x3, 0, z4, xrot, -3 * pi / 4))
    # central
    for i in range(4):
        view_points.append((x3, 0, z3, xrot, i * pi / 2))

    for i in range(len(view_points)):
        x, y, z, xrot, yrot = view_points[i]
        y = get_height(height_map, x, z) * MAX_HEIGHT + 15
        yrot = pi / 2 - yrot
        view_points[i] = (x, y, z, xrot, yrot)

    return view_points


def pcg(
    height_map: List[List[float]],
    points: List[Tuple[float, float]],
    edges: List[Tuple[List[Tuple[int, int]], List[int]]],
    areas: List[Tuple[int, List[int]]],
    infrustructure: Tuple[List[Tuple[int, int]], List[Tuple[int, int]], dict],
    parameters: dict,
    data: dict,
    gen_idx: int,
    suffix: str = "",
) -> None:
    """
    points[i]: (x, y)
    edges[i]: (points, info) info: 0: wall, 1: main road, 2: secondary road, 3: type border, 4: lakeside, 5: bridge, 6: wall with hole
    edges[i][0][j]: (point_idx1, point_idx2)
    areas[i]: (type, circuit) type: 1: water, 2: lotus, 3: lake rock, 4: pure land, 5: grass/flower, 6: bush, 7: tree,
                                    8: pavilion, 9: plaza, 10: flowerbed, 11: tree lines, 12: building, 13: hill rock, 14: grass
    areas[i][1][j]: point_idx
    """
    logpath = "logs/" + str(gen_idx) + suffix + ".txt"
    sys.stdout = open(logpath, "a")
    sys.stderr = open(logpath, "a")

    entrance_points, keyunits, edge_used = infrustructure
    entrance_points = [(x * RL, y * RL) for x, y in entrance_points]
    points = [(x * RL, y * RL) for x, y in points]
    (
        lotus_data,
        plant_data,
        lake_rock_data,
        flower_data,
        plantbush_data,
        bush_data,
        bigbush_data,
        rectbush_data,
        bamboo_data,
        hugetree_data,
        bigtree_data,
        pavilion_data,
        building_data,
        hill_rock_data,
        wall_data,
        bridge_data,
        statue_data,
    ) = (
        data["lotus"],
        data["plant"],
        data["lake_rock"],
        data["flower"],
        data["plantbush"],
        data["bush"],
        data["bigbush"],
        data["rectbush"],
        data["bamboo"],
        data["hugetree"],
        data["bigtree"],
        data["pavilion"],
        data["building"],
        data["hill_rock"],
        data["wall"],
        data["bridge"],
        data["statue"],
    )
    terrain_labels = np.zeros((MAP_W, MAP_H, 5))  # 0: none, 1: main road, 2: secondary road, 3: plaza, 4: land
    all_obj_infos = []

    edge2info = {}
    for edge_group in edges:
        edge_idxs, info = edge_group
        for i in range(len(edge_idxs)):
            edge2info[edge_idxs[i]] = info
            edge2info[(edge_idxs[i][1], edge_idxs[i][0])] = info

    print("handle edges:")
    for edge_group in tqdm(edges):
        edge_idxs, info = edge_group
        for i in range(len(edge_idxs) - 1):
            if edge_idxs[i][1] != edge_idxs[i + 1][0]:
                raise ValueError("Edge not connected")
        point_idxs = [idx[0] for idx in edge_idxs] + [edge_idxs[-1][1]]
        point_locs = [points[idx] for idx in point_idxs]
        rand_num = random.random()
        total_length = 0
        for i in range(len(point_locs) - 1):
            total_length += eu_dist(point_locs[i], point_locs[i + 1])
        if 3 in info:  # type border
            pass
        elif 5 in info:  # bridge
            longbridge_data, zigzagbridge_data = find_data("Bridge", bridge_data), find_data(
                "Zigzag_Bridge", bridge_data
            )
            if (2 in info) and rand_num < 0.7:
                obj_infos = build_zigzagbridge(height_map, point_locs, zigzagbridge_data)
                all_obj_infos.append((obj_infos, "Zigzag_Bridge"))
            else:
                obj_infos, road_idxs = build_bridge(height_map, point_locs, longbridge_data)
                all_obj_infos.append((obj_infos, "Bridge"))
                for idx in road_idxs:
                    terrain_labels[idx[0]][idx[1]][1] += 1
        elif 6 in info:  # entrance
            wall4_data = find_data("Wall_400x300", wall_data)
            obj_infos = build_entrance(height_map, point_locs, wall4_data, entrance_points)
            all_obj_infos.append((obj_infos, "Wall_400x300"))
        elif 0 in info:  # wall
            wall4_data = find_data("Wall_400x300", wall_data)
            obj_infos = build_wall(height_map, point_locs, wall4_data)
            all_obj_infos.append((obj_infos, "Wall_400x300"))
        elif (1 in info) or (2 in info):  # road
            poly = None
            for i in range(len(point_locs) - 1):
                sp, ep = p(point_locs[i]), p(point_locs[i + 1])
                tang = p(ep[0] - sp[0], ep[1] - sp[1])
                tang = tang / np.linalg.norm(tang)
                norm = p(ep[1] - sp[1], sp[0] - ep[0])
                norm = norm / np.linalg.norm(norm)
                road_width = MAIN_ROAD_WIDTH if (1 in info) else SUB_ROAD_WIDTH
                new_poly = Polygon(
                    [
                        sp + norm * road_width / 2 - tang * 1e-3,
                        sp - norm * road_width / 2 - tang * 1e-3,
                        ep - norm * road_width / 2 + tang * 1e-3,
                        ep + norm * road_width / 2 + tang * 1e-3,
                    ]
                )
                if poly is None:
                    poly = new_poly
                else:
                    poly = poly.union(new_poly)
            poly = poly.simplify(0, False)
            labeling_area(terrain_labels, height_map, poly, 1 if (1 in info) else 2, 1)

    lakeside_buffer = 3
    bridge_buffer = 3
    wall_buffer = 0.5

    print("handle areas:")
    for area in tqdm(areas):
        area_type, circuit = area
        if len(circuit) < 3:
            # for i in range(1000):
            #     print("-----------------")
            # print(circuit)
            # for edge_group in edges:
            #     edge_idxs, info = edge_group
            #     print(edge_idxs)
            continue
        poly = Polygon([points[idx] for idx in circuit])
        buffer = None
        for i in range(len(circuit)):
            idx1, idx2 = circuit[i], circuit[(i + 1) % len(circuit)]
            if idx1 == idx2:
                continue
            p1, p2 = p(points[idx1]), p(points[idx2])
            if (idx1, idx2) in edge2info:
                edge_info = edge2info[(idx1, idx2)]
                if 3 in edge_info:
                    continue
                if 5 in edge_info:
                    buffer = bridge_buffer
                elif 0 in edge_info:
                    buffer = wall_buffer
                elif 4 in edge_info:
                    buffer = lakeside_buffer
                elif 1 in edge_info:
                    buffer = MAIN_ROAD_WIDTH / 2
                else:
                    buffer = SUB_ROAD_WIDTH
            else:
                continue
            tang = p2 - p1
            tang = tang / np.linalg.norm(tang)
            norm = p(p2[1] - p1[1], p1[0] - p2[0])
            norm = norm / np.linalg.norm(norm)
            if (tang is None) or (norm is None):
                continue
            diff_poly = Polygon(
                [
                    p1 + norm * buffer - tang * 0.5,
                    p1 - norm * buffer - tang * 0.5,
                    p2 - norm * buffer + tang * 0.5,
                    p2 + norm * buffer + tang * 0.5,
                ]
            )
            poly = poly.difference(diff_poly)
        poly = poly.simplify(0, False)
        polys = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
        for poly in polys:
            if poly.area < 1:
                continue

            if area_type == 2:  # lotus
                obj_infos = add_lotus(height_map, poly, lotus_data)
                all_obj_infos += obj_infos
            elif area_type == 3:  # lake rock
                obj_infos = add_lakerock(height_map, poly, lake_rock_data)
                all_obj_infos += obj_infos
            elif area_type == 4:  # land
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_rockmaze(height_map, poly, hill_rock_data, lake_rock_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 5:  # bushes
                obj_infos = add_bushes(height_map, poly, rectbush_data, bigbush_data)
                obj_infos2 = add_few_trees(height_map, poly, bigtree_data)
                all_obj_infos += obj_infos
                all_obj_infos += obj_infos2
            elif area_type == 6:  # bamboos
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_bamboos(height_map, poly, bamboo_data)
                all_obj_infos += obj_infos
            elif area_type == 7:  # tree
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_trees(height_map, poly, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 8:  # pavilion
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_pavilion(terrain_labels, height_map, poly, pavilion_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 9:  # plaza
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = add_hugetree(height_map, poly, hugetree_data)
                all_obj_infos += obj_infos
            elif area_type == 10:  # flower/bush beds
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = None
                if random.random() < 0.4 and poly.area < 800:
                    obj_infos = add_plantbeds(terrain_labels, height_map, poly, flower_data, bush_data)
                elif random.random() < 0.6:
                    obj_infos = add_building(height_map, poly, building_data, statue_data, bigtree_data)
                else:
                    obj_infos = add_rockmaze(height_map, poly, hill_rock_data, lake_rock_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 11:  # tree lines
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = add_treelines(terrain_labels, height_map, poly, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 12:  # building
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = add_building(height_map, poly, building_data, statue_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 13:  # hill rock
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_hillrock(height_map, poly, hill_rock_data)
                obj_infos2 = add_hugetree(height_map, poly, hugetree_data)
                all_obj_infos += obj_infos
                all_obj_infos += obj_infos2
            elif area_type == 14:  # plants and bushes
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_plants(height_map, poly, plant_data, plantbush_data)
                obj_infos2 = add_few_trees(height_map, poly, bigtree_data)
                all_obj_infos += obj_infos
                all_obj_infos += obj_infos2

    view_points = decide_view_points(height_map, keyunits)
    output_height_map(height_map, MAP_W, MAP_H, "height_map_" + str(gen_idx), suffix)
    output_label_map(terrain_labels, MAP_W, MAP_H, "label_map_" + str(gen_idx), suffix)
    output_scene(all_obj_infos, view_points, data, gen_idx, suffix)


def add_rockmaze_random(
    height_map: List[List[float]],
    poly: Polygon,
    hill_rock_data: List[dict],
    lake_rock_data: List[dict],
    tree_data: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    hillrock_size_list = [(dat["size"][0], dat["size"][2]) for dat in hill_rock_data]
    lakerock_size_list = [(dat["size"][0], dat["size"][2]) for dat in lake_rock_data]
    tree_size_list = [(dat["size"][0], dat["size"][2]) for dat in tree_data]
    lakerock_point_and_types = random_placing(poly, lakerock_size_list, 0.05, True)
    hillrock_point_and_types = random_placing(poly, hillrock_size_list, 0.1, True)
    tree_point_and_types = random_placing(poly, tree_size_list, 0.2, True)
    hillrock_obj_infos_dict, lakerock_obj_infos_dict, tree_obj_infos_dict = {}, {}, {}
    for typ_idx in range(len(hill_rock_data)):
        hillrock_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(lake_rock_data)):
        lakerock_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(tree_data)):
        tree_obj_infos_dict[typ_idx] = []

    for point_and_type in hillrock_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT - random.uniform(0.2, 0.4)
        hillrock_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in lakerock_point_and_types:
        x, z, typ = point_and_type
        x += random.uniform(-1, 1)
        z += random.uniform(-1, 1)
        y = get_height(height_map, x, z) * MAX_HEIGHT - random.uniform(0.1, 0.3)
        lakerock_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in tree_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        tree_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    obj_infos_list = []
    for typ_idx in hillrock_obj_infos_dict:
        obj_infos_list.append((hillrock_obj_infos_dict[typ_idx], hill_rock_data[typ_idx]["name"]))
    for typ_idx in lakerock_obj_infos_dict:
        obj_infos_list.append((lakerock_obj_infos_dict[typ_idx], lake_rock_data[typ_idx]["name"]))
    for typ_idx in tree_obj_infos_dict:
        obj_infos_list.append((tree_obj_infos_dict[typ_idx], tree_data[typ_idx]["name"]))
    return obj_infos_list


def add_bushes_random(
    height_map: List[List[float]], poly: Polygon, rectbush_datas: List[dict], bigbush_datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    rectbush_size_list = [(dat["size"][0], dat["size"][2]) for dat in rectbush_datas]
    bigbush_size_list = [(dat["size"][0], dat["size"][2]) for dat in bigbush_datas]
    poly_area = poly.area
    point_and_types = None
    typeflag = 0
    point_and_types = random_placing(poly, rectbush_size_list, 0.3, True)
    obj_infos_dict = {}
    length = len(rectbush_datas) if typeflag == 0 else len(bigbush_datas)
    for typ_idx in range(length):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        if typeflag == 0:
            obj_infos_list.append((obj_infos_dict[typ_idx], rectbush_datas[typ_idx]["name"]))
        else:
            obj_infos_list.append((obj_infos_dict[typ_idx], bigbush_datas[typ_idx]["name"]))
    return obj_infos_list


def add_bamboos_random(
    height_map: List[List[float]], poly: Polygon, datas: List[dict]
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    size_list = [(dat["size"][0], dat["size"][2]) for dat in datas]
    poly_area = poly.area
    point_and_types = random_placing(poly, size_list, 0.6, True)
    obj_infos_dict = {}
    for typ_idx in range(len(datas)):
        obj_infos_dict[typ_idx] = []

    for point_and_type in point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))

    obj_infos_list = []
    for typ_idx in obj_infos_dict:
        obj_infos_list.append((obj_infos_dict[typ_idx], datas[typ_idx]["name"]))
    return obj_infos_list


def add_pavilion_random(
    terrain_labels: List[List[List[float]]],
    height_map: List[List[float]],
    poly: Polygon,
    pav_datas: List[dict],
    tree_datas: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    pav_size_list = [(dat["size"][0], dat["size"][2]) for dat in pav_datas]
    tree_size_list = [(dat["size"][0], dat["size"][2]) for dat in tree_datas]
    pav_point_and_types = random_placing(poly, pav_size_list, 0.25, True)
    tree_point_and_types = random_placing(poly, tree_size_list, 0.5, True)
    pav_obj_infos_dict, tree_obj_infos_dict = {}, {}
    for typ_idx in range(len(pav_datas)):
        pav_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(tree_datas)):
        tree_obj_infos_dict[typ_idx] = []

    for point_and_type in pav_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        pav_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in tree_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        tree_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    obj_infos_list = []
    for typ_idx in pav_obj_infos_dict:
        obj_infos_list.append((pav_obj_infos_dict[typ_idx], pav_datas[typ_idx]["name"]))
    for typ_idx in tree_obj_infos_dict:
        obj_infos_list.append((tree_obj_infos_dict[typ_idx], tree_datas[typ_idx]["name"]))

    return obj_infos_list


def add_plantbeds_random(
    terrain_labels: List[List[List[float]]],
    height_map: List[List[float]],
    poly: Polygon,
    flower_datas: List[dict],
    bush_datas: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:
    flower_size_list, bush_size_list = [(dat["size"][0], dat["size"][2]) for dat in flower_datas], [
        (dat["size"][0], dat["size"][2]) for dat in bush_datas
    ]
    flower_point_and_types = random_placing(poly, flower_size_list, 0.3, True)
    bush_point_and_types = random_placing(poly, bush_size_list, 0.3, True)
    flower_obj_infos_dict, bush_obj_infos_dict = {}, {}
    for typ_idx in range(len(flower_datas)):
        flower_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(bush_datas)):
        bush_obj_infos_dict[typ_idx] = []

    for point_and_type in flower_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        flower_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in bush_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        bush_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    obj_infos_list = []
    for typ_idx in flower_obj_infos_dict:
        obj_infos_list.append((flower_obj_infos_dict[typ_idx], flower_datas[typ_idx]["name"]))
    for typ_idx in bush_obj_infos_dict:
        obj_infos_list.append((bush_obj_infos_dict[typ_idx], bush_datas[typ_idx]["name"]))
    return obj_infos_list


def add_building_random(
    height_map: List[List[float]],
    poly: Polygon,
    buid_datas: List[dict],
    statue_datas: List[dict],
    tree_datas: List[dict],
) -> List[Tuple[List[Tuple[float, float, float, float]], str]]:  # TODO
    buid_size_list = [(dat["size"][0], dat["size"][2]) for dat in buid_datas]
    statue_size_list = [(dat["size"][0], dat["size"][2]) for dat in statue_datas]
    tree_size_list = [(dat["size"][0], dat["size"][2]) for dat in tree_datas]
    buid_point_and_types = random_placing(poly, buid_size_list, 0.3, True)
    statue_point_and_types = random_placing(poly, statue_size_list, 0.02, True)
    tree_point_and_types = random_placing(poly, tree_size_list, 0.05, True)
    buid_obj_infos_dict, statue_obj_infos_dict, tree_obj_infos_dict = {}, {}, {}
    for typ_idx in range(len(buid_datas)):
        buid_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(statue_datas)):
        statue_obj_infos_dict[typ_idx] = []
    for typ_idx in range(len(tree_datas)):
        tree_obj_infos_dict[typ_idx] = []

    for point_and_type in buid_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        buid_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in statue_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        statue_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    for point_and_type in tree_point_and_types:
        x, z, typ = point_and_type
        y = get_height(height_map, x, z) * MAX_HEIGHT
        tree_obj_infos_dict[typ].append((x, y, z, random.uniform(0, 2 * pi)))
    obj_infos_list = []
    for typ_idx in buid_obj_infos_dict:
        obj_infos_list.append((buid_obj_infos_dict[typ_idx], buid_datas[typ_idx]["name"]))
    for typ_idx in statue_obj_infos_dict:
        obj_infos_list.append((statue_obj_infos_dict[typ_idx], statue_datas[typ_idx]["name"]))
    for typ_idx in tree_obj_infos_dict:
        obj_infos_list.append((tree_obj_infos_dict[typ_idx], tree_datas[typ_idx]["name"]))
    return obj_infos_list


def pcg_random(
    height_map: List[List[float]],
    points: List[Tuple[float, float]],
    edges: List[Tuple[List[Tuple[int, int]], List[int]]],
    areas: List[Tuple[int, List[int]]],
    infrustructure: Tuple[List[Tuple[int, int]], List[Tuple[Tuple[int, int], Tuple[int, int]]], dict],
    parameters: dict,
    data: dict,
    gen_idx: int,
    suffix: str = "",
) -> None:
    """
    points[i]: (x, y)
    edges[i]: (points, info) info: 0: wall, 1: main road, 2: secondary road, 3: type border, 4: lakeside, 5: bridge, 6: wall with hole
    edges[i][0][j]: (point_idx1, point_idx2)
    areas[i]: (type, circuit) type: 1: water, 2: lotus, 3: lake rock, 4: pure land, 5: grass/flower, 6: bush, 7: tree,
                                    8: pavilion, 9: plaza, 10: flowerbed, 11: tree lines, 12: building, 13: hill rock, 14: grass
    areas[i][1][j]: point_idx
    """
    logpath = "logs/" + str(gen_idx) + suffix + ".txt"
    sys.stdout = open(logpath, "a")
    sys.stderr = open(logpath, "a")

    entrance_points, keyunits, edge_used = infrustructure
    entrance_points = [(x * RL, y * RL) for x, y in entrance_points]
    points = [(x * RL, y * RL) for x, y in points]
    (
        lotus_data,
        plant_data,
        lake_rock_data,
        flower_data,
        plantbush_data,
        bush_data,
        bigbush_data,
        rectbush_data,
        bamboo_data,
        hugetree_data,
        bigtree_data,
        pavilion_data,
        building_data,
        hill_rock_data,
        wall_data,
        bridge_data,
        statue_data,
    ) = (
        data["lotus"],
        data["plant"],
        data["lake_rock"],
        data["flower"],
        data["plantbush"],
        data["bush"],
        data["bigbush"],
        data["rectbush"],
        data["bamboo"],
        data["hugetree"],
        data["bigtree"],
        data["pavilion"],
        data["building"],
        data["hill_rock"],
        data["wall"],
        data["bridge"],
        data["statue"],
    )
    terrain_labels = np.zeros((MAP_W, MAP_H, 5))  # 0: none, 1: main road, 2: secondary road, 3: plaza, 4: land
    all_obj_infos = []

    edge2info = {}
    for edge_group in edges:
        edge_idxs, info = edge_group
        for i in range(len(edge_idxs)):
            edge2info[edge_idxs[i]] = info
            edge2info[(edge_idxs[i][1], edge_idxs[i][0])] = info

    print("handle edges:")
    for edge_group in tqdm(edges):
        edge_idxs, info = edge_group
        for i in range(len(edge_idxs) - 1):
            if edge_idxs[i][1] != edge_idxs[i + 1][0]:
                raise ValueError("Edge not connected")
        point_idxs = [idx[0] for idx in edge_idxs] + [edge_idxs[-1][1]]
        point_locs = [points[idx] for idx in point_idxs]
        rand_num = random.random()
        total_length = 0
        for i in range(len(point_locs) - 1):
            total_length += eu_dist(point_locs[i], point_locs[i + 1])
        if 3 in info:  # type border
            pass
        elif 5 in info:  # bridge
            longbridge_data, zigzagbridge_data = find_data("Bridge", bridge_data), find_data(
                "Zigzag_Bridge", bridge_data
            )
            if (2 in info) and rand_num < 0.7:
                obj_infos = build_zigzagbridge(height_map, point_locs, zigzagbridge_data)
                all_obj_infos.append((obj_infos, "Zigzag_Bridge"))
            else:
                obj_infos, road_idxs = build_bridge(height_map, point_locs, longbridge_data)
                all_obj_infos.append((obj_infos, "Bridge"))
                for idx in road_idxs:
                    terrain_labels[idx[0]][idx[1]][1] += 1
        elif 6 in info:  # entrance
            wall4_data = find_data("Wall_400x300", wall_data)
            obj_infos = build_entrance(height_map, point_locs, wall4_data, entrance_points)
            all_obj_infos.append((obj_infos, "Wall_400x300"))
        elif 0 in info:  # wall
            wall4_data = find_data("Wall_400x300", wall_data)
            obj_infos = build_wall(height_map, point_locs, wall4_data)
            all_obj_infos.append((obj_infos, "Wall_400x300"))
        elif (1 in info) or (2 in info):  # road
            poly = None
            for i in range(len(point_locs) - 1):
                sp, ep = p(point_locs[i]), p(point_locs[i + 1])
                tang = p(ep[0] - sp[0], ep[1] - sp[1])
                tang = tang / np.linalg.norm(tang)
                norm = p(ep[1] - sp[1], sp[0] - ep[0])
                norm = norm / np.linalg.norm(norm)
                road_width = MAIN_ROAD_WIDTH if (1 in info) else SUB_ROAD_WIDTH
                new_poly = Polygon(
                    [
                        sp + norm * road_width / 2 - tang * 1e-3,
                        sp - norm * road_width / 2 - tang * 1e-3,
                        ep - norm * road_width / 2 + tang * 1e-3,
                        ep + norm * road_width / 2 + tang * 1e-3,
                    ]
                )
                if poly is None:
                    poly = new_poly
                else:
                    poly = poly.union(new_poly)
            poly = poly.simplify(0, False)
            labeling_area(terrain_labels, height_map, poly, 1 if (1 in info) else 2, 1)

    lakeside_buffer = 3
    bridge_buffer = 3
    wall_buffer = 0.5

    print("handle areas:")
    for area in tqdm(areas):
        area_type, circuit = area
        if len(circuit) < 3:
            continue
        poly = Polygon([points[idx] for idx in circuit])
        buffer = None
        for i in range(len(circuit)):
            idx1, idx2 = circuit[i], circuit[(i + 1) % len(circuit)]
            if idx1 == idx2:
                continue
            p1, p2 = p(points[idx1]), p(points[idx2])
            if (idx1, idx2) in edge2info:
                edge_info = edge2info[(idx1, idx2)]
                if 3 in edge_info:
                    continue
                if 5 in edge_info:
                    buffer = bridge_buffer
                elif 0 in edge_info:
                    buffer = wall_buffer
                elif 4 in edge_info:
                    buffer = lakeside_buffer
                elif 1 in edge_info:
                    buffer = MAIN_ROAD_WIDTH / 2
                else:
                    buffer = SUB_ROAD_WIDTH
            else:
                continue
            tang = p2 - p1
            tang = tang / np.linalg.norm(tang)
            norm = p(p2[1] - p1[1], p1[0] - p2[0])
            norm = norm / np.linalg.norm(norm)
            if (tang is None) or (norm is None):
                continue
            diff_poly = Polygon(
                [
                    p1 + norm * buffer - tang * 0.5,
                    p1 - norm * buffer - tang * 0.5,
                    p2 - norm * buffer + tang * 0.5,
                    p2 + norm * buffer + tang * 0.5,
                ]
            )
            poly = poly.difference(diff_poly)
        poly = poly.simplify(0, False)
        polys = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
        for poly in polys:
            if poly.area < 1:
                continue

            if area_type == 2:  # lotus
                obj_infos = add_lotus(height_map, poly, lotus_data)
                all_obj_infos += obj_infos
            elif area_type == 3:  # lake rock
                obj_infos = add_lakerock(height_map, poly, lake_rock_data)
                all_obj_infos += obj_infos
            elif area_type == 4:  # land
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_rockmaze_random(height_map, poly, hill_rock_data, lake_rock_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 5:  # bushes
                obj_infos = add_bushes_random(height_map, poly, rectbush_data, bigbush_data)
                all_obj_infos += obj_infos
            elif area_type == 6:  # bamboos
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_bamboos_random(height_map, poly, bamboo_data)
                all_obj_infos += obj_infos
            elif area_type == 7:  # tree
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_trees(height_map, poly, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 8:  # pavilion
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_pavilion_random(terrain_labels, height_map, poly, pavilion_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 9:  # plaza
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = add_hugetree(height_map, poly, hugetree_data)
                all_obj_infos += obj_infos
            elif area_type == 10:  # flower/bush beds
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = None
                if random.random() < 0.5:
                    obj_infos = add_plantbeds_random(terrain_labels, height_map, poly, flower_data, bush_data)
                else:
                    obj_infos = add_rockmaze_random(height_map, poly, hill_rock_data, lake_rock_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 11:  # tree lines
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = add_trees(height_map, poly, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 12:  # building
                labeling_area(terrain_labels, height_map, poly, 3, 0.5)
                obj_infos = add_building_random(height_map, poly, building_data, statue_data, bigtree_data)
                all_obj_infos += obj_infos
            elif area_type == 13:  # hill rock
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_hillrock(height_map, poly, hill_rock_data)
                all_obj_infos += obj_infos
            elif area_type == 14:  # plants and bushes
                labeling_area(terrain_labels, height_map, poly, 4, 1)
                obj_infos = add_plants(height_map, poly, plant_data, plantbush_data)
                all_obj_infos += obj_infos

    view_points = decide_view_points(height_map, keyunits)
    output_height_map(height_map, MAP_W, MAP_H, "height_map_" + str(gen_idx), suffix)
    output_label_map(terrain_labels, MAP_W, MAP_H, "label_map_" + str(gen_idx), suffix)
    output_scene(all_obj_infos, view_points, data, gen_idx, suffix)
