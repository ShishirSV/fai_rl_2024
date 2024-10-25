from poke_env.environment.move_category import MoveCategory

def calculate_damage(move, attacker, defender, pessimistic, is_bot_turn):
    if move is None or move.category == MoveCategory.STATUS:
        return 0

    damage = move.base_power
    ratio = calculate_stat_ratio(move, attacker, defender, is_bot_turn)
    level_multiplier = ((2 * attacker.level) / 5) + 2

    damage *= ratio * level_multiplier
    damage = (damage / 50) + 2

    if pessimistic:
        damage *= 0.85

    if move.type in [attacker.type_1, attacker.type_2]:
        damage *= 1.5

    type_multiplier = defender.damage_multiplier(move)
    return damage * type_multiplier


def calculate_stat_ratio(move, attacker, defender, is_bot_turn):
    if move.category == MoveCategory.PHYSICAL:
        return calculate_physical_ratio(attacker, defender, is_bot_turn)
    elif move.category == MoveCategory.SPECIAL:
        return calculate_special_ratio(attacker, defender, is_bot_turn)


def calculate_physical_ratio(attacker, defender, is_bot_turn):
    if is_bot_turn:
        attack = attacker.stats["atk"]
        defense = defender.base_stats["def"] * 2 + 36
    else:
        defense = defender.stats["def"]
        attack = attacker.base_stats["atk"] * 2 + 36

    defense = ((defense * defender.level) / 100) + 5
    attack = ((attack * attacker.level) / 100) + 5
    return attack / defense


def calculate_special_ratio(attacker, defender, is_bot_turn):
    if is_bot_turn:
        spatk = attacker.stats["spa"]
        spdef = defender.base_stats["spd"] * 2 + 36
    else:
        spdef = defender.stats["spd"]
        spatk = attacker.base_stats["spa"] * 2 + 36

    spdef = ((spdef * defender.level) / 100) + 5
    spatk = ((spatk * attacker.level) / 100) + 5
    return spatk / spdef


def opponent_can_outspeed(my_pokemon, opponent_pokemon):
    my_speed = my_pokemon.stats["spe"]
    opponent_speed = (opponent_pokemon.base_stats["spe"] * 2 + 52) * opponent_pokemon.level / 100 + 5
    return opponent_speed > my_speed


def calculate_total_HP(pokemon, is_dynamaxed):
    HP = pokemon.base_stats["hp"] * 2 + 36
    HP = ((HP * pokemon.level) / 100) + pokemon.level + 10
    return HP * 2 if is_dynamaxed else HP


def get_defensive_type_multiplier(my_pokemon, opponent_pokemon):
    first_multiplier = my_pokemon.damage_multiplier(opponent_pokemon.type_1)
    second_multiplier = my_pokemon.damage_multiplier(opponent_pokemon.type_2) if opponent_pokemon.type_2 else 1
    return max(first_multiplier, second_multiplier)
