from cvxopt import matrix, solvers
import cvxpy as cp
from dataclasses import dataclass
from enum import Enum
from gekko import GEKKO
import json
from mystic.constraints import integers
from mystic.solvers import fmin_powell
from mystic.symbolic import generate_conditions, generate_constraint, generate_penalty, generate_solvers, simplify
import numpy as np
from scipy import optimize

# User-inputted variables
base_strength = 50 + 40 + 104 + 40  # Profile + armor + weapon + potion
base_crit_chance = 38 + 24 + 1  # Profile + armor + weapon
base_crit_damage = 52 + 72 + 95 + 50  # Profile + armor + weapon + enchants
combat_level = 16  # Skill
weapon_damage = 100  # Weapon
weapon_bonus = 0  # Weapon
enchants = 1 + 0.25  # FirstStrikeIV + SharpnessV
armor_bonus = 1  # Armor
target_crit_chance = 100 - 20  # Perfect - potion


class Rarity(Enum):
    COMMON = {
        "value": 0,
        "name": "Common"
    }
    UNCOMMON = {
        "value": 1,
        "name": "Uncommon"
    }
    RARE = {
        "value": 2,
        "name": "Rare"
    }
    EPIC = {
        "value": 3,
        "name": "Epic"
    }


accessories = {
    Rarity.COMMON: 12,
    Rarity.UNCOMMON: 14,
    Rarity.RARE: 17,
    Rarity.EPIC: 4
}

"""
Damage Multipliers
------------------
Melee:      (5 + weaponDmg)
Additional: (1 + ([combatLevel <= 50] * 0.04 + [combatLevel > 50] * 0.01) + enchants + weaponBonus)
Armor:      (armorBonus)
Strength:   (1 + strength / 100)
Crit:       (1 + critDamage / 100)

Crit Probability
----------------
strength * (base * critDamage * critChance + base * (1 - critChance))
= strength * (base * critDamage * critChance + base - base * critChance)
= strength * (base * critChance * critDamage - base * critChance + base)
= base * strength * critChance * critDamage - base * strength * critChance + base * strength

Optimization Example Tables
---------------------------
https://docs.google.com/spreadsheets/d/1P4Zcq547upT1NR8OdrC2Gt3afH-mIzFZgGfNhZAEXoI

Optimization Example #1 (2x)
----------------------------
= (C x(11) x(12) x(13))/1000000
+ (C x(11) x(12) x(23))/1000000
+ (C x(11) x(13) x(22))/1000000
+ (C x(11) x(22) x(23))/1000000
+ (C x(12) x(13) x(21))/1000000
+ (C x(12) x(21) x(23))/1000000
+ (C x(13) x(21) x(22))/1000000
+ (C x(21) x(22) x(23))/1000000
+ (C x(11) x(13))/10000
+ (C x(11) x(23))/10000
+ (C x(12) x(13))/10000
+ (C x(12) x(23))/10000
+ (C x(13) x(21))/10000
+ (C x(13) x(22))/10000
+ (C x(21) x(23))/10000
+ (C x(22) x(23))/10000
+ (C x(11))/100
+ (C x(13))/100
+ (C x(21))/100
+ (C x(23))/100
+ (C)

Optimization Problem #2 (3x)
= (C x(31) x(32) x(33))/1000000
+ (C x(21) x(32) x(33))/1000000
+ (C x(11) x(32) x(33))/1000000
+ (C x(22) x(31) x(33))/1000000
+ (C x(12) x(31) x(33))/1000000
+ (C x(21) x(22) x(33))/1000000
+ (C x(11) x(22) x(33))/1000000
+ (C x(12) x(21) x(33))/1000000
+ (C x(11) x(12) x(33))/1000000
+ (C x(23) x(31) x(32))/1000000
+ (C x(13) x(31) x(32))/1000000
+ (C x(21) x(23) x(32))/1000000
+ (C x(11) x(23) x(32))/1000000
+ (C x(13) x(21) x(32))/1000000
+ (C x(11) x(13) x(32))/1000000
+ (C x(22) x(23) x(31))/1000000
+ (C x(12) x(23) x(31))/1000000
+ (C x(13) x(22) x(31))/1000000
+ (C x(12) x(13) x(31))/1000000
+ (C x(21) x(22) x(23))/1000000
+ (C x(11) x(22) x(23))/1000000
+ (C x(11) x(12) x(23))/1000000
+ (C x(12) x(21) x(23))/1000000
+ (C x(13) x(21) x(22))/1000000
+ (C x(11) x(13) x(22))/1000000
+ (C x(11) x(12) x(13))/1000000
+ (C x(12) x(13) x(21))/1000000
+ (C x(12) x(33))/10000
+ (C x(11) x(33))/10000
+ (C x(21) x(33))/10000
+ (C x(22) x(33))/10000
+ (C x(31) x(33))/10000
+ (C x(32) x(33))/10000
+ (C x(23) x(32))/10000
+ (C x(13) x(32))/10000
+ (C x(23) x(31))/10000
+ (C x(13) x(31))/10000
+ (C x(22) x(23))/10000
+ (C x(21) x(23))/10000
+ (C x(12) x(23))/10000
+ (C x(11) x(23))/10000
+ (C x(13) x(22))/10000
+ (C x(13) x(21))/10000
+ (C x(12) x(13))/10000
+ (C x(11) x(13))/10000
+ 1/100 C x(13)
+ 1/100 C x(11)
+ 1/100 C x(21)
+ 1/100 C x(23)
+ 1/100 C x(31)
+ 1/100 C x(33)
+ C
"""


@dataclass
class Reforge:
    rarity: Rarity
    name: str
    strength: int
    critDamage: int
    critChance: int


accessory_reforges = [
    Reforge(Rarity.COMMON, "itchy", 1, 3, 0),
    Reforge(Rarity.COMMON, "forceful", 4, 0, 0),
    Reforge(Rarity.COMMON, "hurtful", 0, 4, 0),
    Reforge(Rarity.COMMON, "superior", 2, 2, 0),
    Reforge(Rarity.COMMON, "unpleasant", 0, 0, 1),
    Reforge(Rarity.UNCOMMON, "itchy", 1, 4, 0),
    Reforge(Rarity.UNCOMMON, "forceful", 5, 0, 0),
    Reforge(Rarity.UNCOMMON, "hurtful", 0, 5, 0),
    Reforge(Rarity.UNCOMMON, "superior", 3, 2, 0),
    Reforge(Rarity.UNCOMMON, "unpleasant", 0, 0, 1),
    Reforge(Rarity.RARE, "itchy", 1, 5, 0),
    Reforge(Rarity.RARE, "forceful", 7, 0, 0),
    Reforge(Rarity.RARE, "hurtful", 0, 7, 0),
    Reforge(Rarity.RARE, "strong", 3, 3, 0),
    Reforge(Rarity.RARE, "superior", 4, 2, 0),
    Reforge(Rarity.RARE, "unpleasant", 0, 0, 1),
    Reforge(Rarity.EPIC, "itchy", 2, 7, 0),
    Reforge(Rarity.EPIC, "forceful", 10, 0, 0),
    Reforge(Rarity.EPIC, "hurtful", 0, 10, 0),
    Reforge(Rarity.EPIC, "strong", 5, 5, 0),
    Reforge(Rarity.EPIC, "unpleasant", 0, 0, 2)
]

if __name__ == "__main__":
    combat_bonus = combat_level * 0.04 if combat_level <= 50 else 50 * 0.04 + (combat_level - 50) * 0.01
    additional = 1 + combat_bonus + enchants + weapon_bonus
    base = float((5 + weapon_damage) * additional * armor_bonus)

    # gekko
    n = len(accessory_reforges)
    zipped = zip(range(n), accessory_reforges)


    def objective(x_n):
        strength = 1 + (sum(list(map(lambda it: it[1].strength * x_n[it[0]], zipped))) + base_strength) / 100
        crit_chance = 1 + (sum(list(map(lambda it: it[1].critChance * x_n[it[0]], zipped))) + base_crit_chance) / 100
        crit_damage = 1 + (sum(list(map(lambda it: it[1].critDamage * x_n[it[0]], zipped))) + base_crit_damage) / 100
        return base * strength * crit_chance * crit_damage - base * strength * crit_chance + base * strength


    m = GEKKO(remote=False)
    x = m.Array(m.Var, n, lb=0, ub=sum(accessories.values()), integer=True)
    m.Maximize(objective(x))
    for rarity in Rarity:
        to_include = filter(lambda x_n: accessory_reforges[x_n].rarity == rarity, range(n))
        m.Equation(sum(map(lambda x_n: x[x_n], to_include)) == accessories[rarity])
    to_include = filter(lambda x_n: accessory_reforges[x_n].critChance > 0, range(n))
    m.Equation(sum(map(lambda x_n: accessory_reforges[x_n].critChance * x[x_n], to_include))
               == target_crit_chance - base_crit_chance)
    m.options.SOLVER = 1
    m.solve()
    resultDict = {
        "Reforges": {}
    }
    for rarity in Rarity:
        resultDict["Reforges"][rarity.value["name"]] = {}
    for i in range(len(x)):
        if x[i][0] != float(0):
            reforge = accessory_reforges[i]
            resultDict["Reforges"][reforge.rarity.value["name"]][reforge.name] = x[i][0]
    print(json.dumps(resultDict, indent=2))

    # scipy
    """
    n = len(accessory_reforges)
    zipped = zip(range(n), accessory_reforges)


    def objective(x):
        strength = 1 + (sum(list(map(lambda it: it[1].strength * x[it[0]], zipped))) + base_strength) / 100
        crit_chance = 1 + (sum(list(map(lambda it: it[1].critChance * x[it[0]], zipped))) + base_crit_chance) / 100
        crit_damage = 1 + (sum(list(map(lambda it: it[1].critDamage * x[it[0]], zipped))) + base_crit_damage) / 100
        return base * strength * crit_chance * crit_damage - base * strength * crit_chance + base * strength


    equations = []
    for rarity in Rarity:
        to_include = filter(lambda x: accessory_reforges[x].rarity == rarity, range(n))
        equations.append(optimize.LinearConstraint([[accessory_reforges[x].strength for x in range(n)]],
                                                   accessories[rarity], accessories[rarity]))
    to_include = filter(lambda x: accessory_reforges[x].critChance > 0, range(n))
    equations.append(optimize.LinearConstraint([[accessory_reforges[x].critChance for x in range(n)]],
                                               target_crit_chance, target_crit_chance))
    x0 = np.zeros((n,), float)
    bounds = [(0, accessories[accessory_reforges[x].rarity]) for x in range(n)]

    result = optimize.minimize(lambda x: -objective(x), x0=x0, bounds=bounds, method="trust-constr",
                               constraints=equations, options={})
    print(result)
    resultDict = {
        "Damage": result["fun"],
        "Reforges": {}
    }
    for rarity in Rarity:
        resultDict["Reforges"][rarity.value["name"]] = {}
    for i in range(len(result["x"])):
        if result["x"][i] != float(0):
            reforge = accessory_reforges[i]
            resultDict["Reforges"][reforge.rarity.value["name"]][reforge.name] = result["x"][i]
    print(json.dumps(resultDict, indent=2))
    """

    # mystic
    """
    n = len(accessoryReforges)
    zipped = zip(range(n), accessoryReforges)


    def objective(x):
        strength = 1 + (sum(list(map(lambda it: it[1].strength * x[it[0]], zipped))) + base_strength) / 100
        crit_chance = 1 + (sum(list(map(lambda it: it[1].critChance * x[it[0]], zipped))) + base_crit_chance) / 100
        crit_damage = 1 + (sum(list(map(lambda it: it[1].critDamage * x[it[0]], zipped))) + base_crit_damage) / 100
        return base * strength * crit_chance * crit_damage - base * strength * crit_chance + base * strength


    @integers()
    def round_constraint(x):
        return x


    equations = ""
    for rarity in Rarity:
        toInclude = filter(lambda x: accessoryReforges[x].rarity == rarity, range(n))
        equations += " + ".join(map(lambda x: "x" + str(x), toInclude)) \
                     + " == " \
                     + str(accessories[rarity]) \
                     + "\n"
    toInclude = filter(lambda x: accessoryReforges[x].critChance != 0, range(n))
    equations += " + ".join(map(lambda x: str(accessoryReforges[x].critChance) + ".x" + str(x), toInclude)) \
                 + " == " \
                 + str(targetCritChance - base_crit_chance) \
                 + "\n"
    print(equations)
    bounds = []
    for reforge in accessoryReforges:
        bounds.append((0, accessories[reforge.rarity]))
    pf = generate_penalty(generate_conditions(equations))
    cf = generate_constraint(generate_solvers(simplify(equations)))

    result = fmin_powell(lambda x: -objective(x), x0=n * [0.0], disp=False, bounds=bounds,
                         constraints=round_constraint, penalty=pf, full_output=True, xtol=1e-1, ftol=1e-1)
    resultDict = {
        "Damage": result[1],
        "Reforges": {}
    }
    for rarity in Rarity:
        resultDict["Reforges"][rarity.value["name"]] = {}
    for i in range(len(result[0])):
        if result[0][i] != float(0):
            reforge = accessoryReforges[i]
            resultDict["Reforges"][reforge.rarity.value["name"]][reforge.name] = result[0][i]
    print(json.dumps(resultDict, indent=2))
    """

    # cvxpy
    """
    n = len(accessoryReforges)
    m = n
    p = len(accessories) + 1
    P = np.empty((n, n), float)
    q = np.empty((n,), float)
    G = np.empty((m, n), float)
    h = np.empty((n,), float)
    A = np.empty((p, n), float)
    b = np.empty((p,), float)
    
    for (i, reforgeI) in enumerate(accessoryReforges):
        for (j, reforgeJ) in enumerate(accessoryReforges):
            if i == j:
                P[i][i] = base / 5000 * reforgeI.strength * reforgeI.critDamage
                G[i][i] = -1.0
            else:
                P[i][j] = base / 10000 * (reforgeI.strength * reforgeJ.critDamage
                                          + reforgeJ.strength * reforgeI.critDamage)
                G[i][j] = 0.0
        q[i] = base / 100.0 * (baseCritDamage / 100.0 * reforgeI.strength
                               + baseStrength / 100.0 * reforgeI.critDamage
                               + reforgeI.strength
                               + reforgeI.critDamage)
        for rarity in Rarity:
            if reforgeI.rarity == rarity:
                A[rarity.value][i] = 1.0
            else:
                A[rarity.value][i] = 0.0
        A[p - 1][i] = float(reforgeI.critChance)
    for rarity in Rarity:
        b[rarity.value] = float(accessories[rarity])
    b[p - 1] = float(targetCritChance)
    
    x = cp.Variable((n,))
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P.T @ P) + q.T @ x), [G @ x <= h, A @ x == b])
    prob.solve()
    """

    # cvxopt
    """
    p = []
    q = []
    g = []
    h = len(accessoryReforges) * [0.0]
    a = []
    b = []
    for (col, colX) in enumerate(accessoryReforges):
        pCol = []
        gCol = []
        for (row, rowX) in enumerate(accessoryReforges):
            if col == row:
                pCol.append(base / 5000 * colX.strength * colX.critDamage)
                gCol.append(-1.0)
            else:
                pCol.append(base / 10000 * (
                        colX.strength * rowX.critDamage + rowX.strength * colX.critDamage
                ))
                gCol.append(0.0)
        p.append(pCol)
        g.append(gCol)
        q.append(base / 100.0 * (
                baseCritDamage / 100.0 * colX.strength
                + baseStrength / 100.0 * colX.critDamage
                + colX.strength
                + colX.critDamage
        ))
        aCol = []
        for rarity in Rarity:
            if colX.rarity == rarity:
                aCol.append(1.0)
            else:
                aCol.append(0.0)
        aCol.append(colX.critChance)
        a.append(aCol)
    for rarity in Rarity:
        b.append(float(accessories[rarity]))
    b.append(float(targetCritChance))
    
    sol = solvers.qp(matrix(p), matrix(q), matrix(g), matrix(h), matrix(a), matrix(b), None, "ldl", {
        "kktreg": 1e-1,
        "abstol": 1e-1,
        "reltol": 1e-1,
        "feastol": 1e-1,
    })
    """
