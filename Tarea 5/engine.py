import math, re, json
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

COLS = [*list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")] + ["AA", "AB", "AC"] 
ROWS = list(range(1, 16))
CELL_W = 3
MAX_LEVEL = 27
ZONE_SIZE = 3
COL_INDEX = {c:i for i,c in enumerate(COLS)}
DX4 = [(1,0),(-1,0),(0,1),(0,-1)]

def clamp(v, a, b): return max(a, min(b, v))
def manhattan(a,b): return abs(a[0]-b[0])+abs(a[1]-b[1])

@dataclass(eq=True, unsafe_hash=True)
class Zone:
    kind: str
    top_left: tuple[int, int]
    level: int = 0
    decayed_prev: bool = False

@dataclass
class RoadCell:
    vertical: bool        
    traffic: int = 0
    traffic_prev: int = 0

class City:
    def __init__(self):
        self.cols = len(COLS)
        self.rows = len(ROWS)
        self.grid_occ = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.zones: List[Zone] = []
        self.roads: Dict[Tuple[int,int], RoadCell] = {}
        self.last_tick_decreased_zones: set = set()

    @staticmethod
    def _parse_cell(token: str) -> Tuple[int,int]:
        token = token.strip().upper()
        m = re.fullmatch(r"([A-Z]{1,2})\s*(\d{1,2})", token)
        if not m: 
            raise ValueError("Invalid Cell. Use p.ex. C4, F10, AB7")
        col_lbl, row_lbl = m.group(1), int(m.group(2))
        if col_lbl not in COL_INDEX or row_lbl not in ROWS:
            raise ValueError("Coordenate Out of Range.")
        c = COL_INDEX[col_lbl]
        r = ROWS.index(row_lbl)
        return (c, r)

    def _can_place_zone(self, c0, r0):
        if c0+ZONE_SIZE>self.cols or r0+ZONE_SIZE>self.rows: 
            return False
        for r in range(r0, r0+ZONE_SIZE):
            for c in range(c0, c0+ZONE_SIZE):
                if (c, r) in self.roads: return False
                if self.grid_occ[r][c] is not None: return False
        return True

    def _occupy_zone(self, z: Zone, occupy: bool):
        for r in range(z.top_left[1], z.top_left[1]+ZONE_SIZE):
            for c in range(z.top_left[0], z.top_left[0]+ZONE_SIZE):
                self.grid_occ[r][c] = z if occupy else None

    def add_zone(self, kind: str, cell: str):
        kind = kind.lower()
        if kind not in ("r","c","i"): 
            raise ValueError("Invalid Type Zone: r/c/i")
        c0, r0 = self._parse_cell(cell)
        if not self._can_place_zone(c0, r0):
            raise ValueError("Occupied Zone or Out of Range.")
        z = Zone(kind=kind, top_left=(c0,r0), level=0)
        self.zones.append(z)
        self._occupy_zone(z, True)
        return f"Zone {kind.upper()} in {cell.upper()}"

    def add_road(self, start: str, end: str):
        c0, r0 = self._parse_cell(start)
        if re.fullmatch(r"[A-Z]{1,2}", end.strip(), re.I):
            c1 = COL_INDEX[end.strip().upper()]
            r1 = r0
            if c1 < c0: c0, c1 = c1, c0
            for c in range(c0, c1+1):
                if self.grid_occ[r0][c] is not None: 
                    raise ValueError("Overlapping With Existing Zone.")
            for c in range(c0, c1+1):
                self.roads[(c,r0)] = RoadCell(vertical=False)
            return f"Horizontal Way {start.upper()}–{end.upper()}"
        else:
            endr = int(end)
            r1 = ROWS.index(endr)
            c1 = c0
            if r1 < r0: r0, r1 = r1, r0
            for r in range(r0, r1+1):
                if self.grid_occ[r][c0] is not None:
                    raise ValueError("Overlapping With Existing Zone.")
            for r in range(r0, r1+1):
                self.roads[(c0,r)] = RoadCell(vertical=True)
            return f"Vertical Way {start.upper()}–{endr}"

    def handle_command(self, cmd: str):
        toks = re.split(r"\s+", cmd.strip())
        if not toks: 
            self.tick()
            return "tick"
        t0 = toks[0].lower()
        if t0 == "r" and len(toks)==2:
            return self.add_zone("r", toks[1])
        if t0 == "c" and len(toks)==2:
            return self.add_zone("c", toks[1])
        if t0 == "i" and len(toks)==2:
            return self.add_zone("i", toks[1])
        if t0 == "a" and len(toks)==3:
            return self.add_road(toks[1], toks[2])
        if t0 == "Save" and len(toks)==2:
            return "Use Save Botton."
        if t0 == "Upload" and len(toks)==2:
            return "Use Upload Botton."
        if t0 == "Escape":
            return "Escape"
        raise ValueError("Invalid Command. Use: r|c|i <cell>, a <start> <end>.")

    def _adjacent_road_cells(self, z: Zone):
        cells = set()
        c0, r0 = z.top_left
        for r in range(r0, r0+ZONE_SIZE):
            for c in range(c0, c0+ZONE_SIZE):
                for dx,dy in DX4:
                    cc, rr = c+dx, r+dy
                    if 0 <= cc < self.cols and 0 <= rr < self.rows and (cc,rr) in self.roads:
                        cells.add((cc,rr))
        return list(cells)

    def _zones_within(self, z: Zone, kind_filter=None, maxdist=10):
        center = (z.top_left[0]+1, z.top_left[1]+1)
        out = []
        for other in self.zones:
            if other is z: continue
            if kind_filter and other.kind != kind_filter: continue
            oc = (other.top_left[0]+1, other.top_left[1]+1)
            if manhattan(center, oc) <= maxdist:
                out.append(other)
        return out

    def _compute_votes(self):
        decayed_last = set()
        votes = {}
        for z in self.zones:
            grow = 0
            decay = 0
            adj_roads = self._adjacent_road_cells(z)
            if z.kind == "r":
                grow += len(adj_roads)
                grow += sum(1 for o in self._zones_within(z, "i", 10) if o.level < 18)
                grow += sum(1 for o in self._zones_within(z, "c", 10) if o.level > math.sqrt(z.level))
                decay += sum(2 for rc in adj_roads if self.roads[rc].traffic > 5)
                inds = self._zones_within(z, "i", 10)
                if len(inds) > 10:
                    decay += 2*(len(inds)-10)
                decay += sum(2 for o in self._zones_within(z, "c", 10) if o.decayed_prev)
            elif z.kind == "c":
                decay += sum(2 for rc in adj_roads if self.roads[rc].traffic < self.roads[rc].traffic_prev)
                grow  += sum(1 for rc in adj_roads if self.roads[rc].traffic > 5)
                grow  += len(self._zones_within(z, "r", 10))
                grow  += sum(1 for o in self._zones_within(z, "i", 10) if o.level > math.sqrt(z.level))
                decay += sum(2 for o in self._zones_within(z, "r", 10) if o.decayed_prev)
                decay += sum(2 for o in self._zones_within(z, "i", 10) if o.decayed_prev)
            elif z.kind == "i":
                grow  += len(adj_roads)
                grow  += len(self._zones_within(z, "r", 10))
                grow  += sum(1 for o in self._zones_within(z, "c", 10) if o.level > math.sqrt(z.level))
                decay += sum(2 for o in self._zones_within(z, "r", 10) if o.decayed_prev)
                decay += sum(2 for o in self._zones_within(z, "c", 10) if o.decayed_prev)
                decay += sum(2 for o in self._zones_within(z, "i", 10) if o.decayed_prev)
            votes[z] = (grow, decay)
        return votes

    def _apply_zone_levels(self, votes: Dict[Zone, Tuple[int,int]]):
        self.last_tick_decreased_zones.clear()
        for z,(g,d) in votes.items():
            if g > d: 
                old = z.level
                z.level = clamp(z.level+1, 0, MAX_LEVEL)
                z.decayed_prev = False
            elif d > g:
                old = z.level
                z.level = clamp(z.level-1, 0, MAX_LEVEL)
                z.decayed_prev = True if z.level < old else False
                if z.decayed_prev: self.last_tick_decreased_zones.add(z)
            else:
                z.decayed_prev = False

    def _compute_road_traffic(self):
        for rc, cell in self.roads.items():
            cell.traffic_prev = cell.traffic
        for rc, cell in self.roads.items():
            c,r = rc
            t = 0.0
            for dx,dy in DX4:
                cc, rr = c+dx, r+dy
                if 0 <= cc < self.cols and 0 <= rr < self.rows:
                    occ = self.grid_occ[rr][cc]
                    if isinstance(occ, Zone):
                        t += math.sqrt(max(0, occ.level))
                    elif (cc,rr) in self.roads:
                        t += self.roads[(cc,rr)].traffic
            self.roads[rc].traffic = int(round(t))

    def tick(self):
        votes = self._compute_votes()
        self._apply_zone_levels(votes)
        self._compute_road_traffic()

    def _zone_cell_letters(self, z: Zone):
        total = ZONE_SIZE*ZONE_SIZE*CELL_W
        uppers = clamp(z.level, 0, MAX_LEVEL)
        letters = z.kind * total
        s = []
        for i,ch in enumerate(letters):
            s.append(ch.upper() if i < uppers else ch.lower())
        joined = "".join(s)
        rows = [joined[i*ZONE_SIZE*CELL_W:(i+1)*ZONE_SIZE*CELL_W] for i in range(ZONE_SIZE)]
        return rows

    def render_grid(self) -> str:
        canvas = [[" "]* (2 + self.cols*CELL_W) for _ in range(1 + self.rows)]
        header = [" "] * 2
        for c, lbl in enumerate(COLS):
            header.extend(list(lbl.rjust(CELL_W)))
        canvas[0] = header
        for r, rownum in enumerate(ROWS, start=1):
            line = list(str(rownum).rjust(2))
            line.extend([" "]* (self.cols*CELL_W))
            canvas[r] = line
        for (c,r), road in self.roads.items():
            rr = r+1
            cc = 2 + c*CELL_W
            if road.vertical:
                token = f"|{road.traffic}|" if road.traffic>1 else "| |"
                for i,ch in enumerate(token[:CELL_W]):
                    canvas[rr][cc+i] = ch
            else:
                if road.traffic>1:
                    token = str(road.traffic).rjust(CELL_W)
                else:
                    token = "="*CELL_W
                for i,ch in enumerate(token[:CELL_W]):
                    canvas[rr][cc+i] = ch
        for z in self.zones:
            zr = self._zone_cell_letters(z)
            for i in range(ZONE_SIZE):
                rr = 1 + z.top_left[1] + i
                cc = 2 + z.top_left[0]*CELL_W
                for j in range(ZONE_SIZE*CELL_W):
                    ch = zr[i][j]
                    if (canvas[rr][cc+j] in ["=","|"," ",":"] or canvas[rr][cc+j].isdigit()):
                        canvas[rr][cc+j] = ch
        for (c,r), road in self.roads.items():
            rr = r+1
            cc = 2 + c*CELL_W
            if road.vertical:
                segment = "".join(canvas[rr][cc:cc+CELL_W])
                if segment[0]=="|" and segment[2]=="|" and (segment[1].isdigit() or segment[1]==" "):
                    continue
            else:
                segment = "".join(canvas[rr][cc:cc+CELL_W])
                if segment.strip("=")=="" or segment.strip().isdigit():
                    continue
        for (c,r), cell in self.roads.items():
            if (c+1,r) in self.roads and self.roads[(c+1,r)].vertical:
                rr = r+1
                cc = 2 + c*CELL_W
                for i,ch in enumerate(":+:"):
                    canvas[rr][cc+i] = ch
        return "\n".join("".join(row) for row in canvas)

    def to_dict(self):
        return {
            "Zones": [{"k":z.kind,"c":z.top_left[0],"r":z.top_left[1],"lv":z.level} for z in self.zones],
            "Roads": [{"c":c,"r":r,"v":rc.vertical,"t":rc.traffic} for (c,r),rc in self.roads.items()]
        }

    @classmethod
    def from_dict(cls, d):
        city = cls()
        for z in d.get("Zones", []):
            zobj = Zone(kind=z["k"], top_left=(z["c"], z["r"]), level=int(z.get("lv",0)))
            city.zones.append(zobj)
            city._occupy_zone(zobj, True)
        for rd in d.get("Roads", []):
            city.roads[(rd["c"], rd["r"])] = RoadCell(vertical=bool(rd["v"]), traffic=int(rd.get("t",0)))
        return city