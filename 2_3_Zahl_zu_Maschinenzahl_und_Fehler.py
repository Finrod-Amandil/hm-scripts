from decimal import Decimal
import math

"""Für Aufgaben der Art
    'Stellen Sie die Zahl x = √3 korrekt gerundet als Maschinenzahl x˜ in einer Fliesskomma-Arithmetik mit 5
    Binärstellen dar, und geben Sie den relativen Fehler von x˜ im Dezimalformat an.'
    
    @version: 1.0, 23.01.2021
    @author: zahlesev@students.zhaw.ch
"""

"""==================== INPUT ===================="""
x = math.sqrt(3)  # Gleitkommazahl im Dezimalsystem (positiv oder negativ)
digitCount = 5    # Anzahl Fliesskommastellen / Mantissestellen
base = 2          # Basis (Binär = 2, Dezimal = 10, Hexadezimal = 16, alle Basen von 2 bis 16 funktionieren)
"""==============================================="""

"""
Rounds up a set of digits with a given base to the next higher possible number
"""
def round_up(digits, base):
    for i in range(len(digits)):
        digit = digits[-(i+1)]

        # No wrap over to next digit
        if digit <= base - 2:
            digits[-(i+1)] = digits[-(i+1)] + 1
            return digits

        # Wrap over to next digit
        digits[-(i+1)] = 0

    # If program flow has not returned after for-loop, all digits have overflowed --> add additional digit
    digits.insert(0, 1)
    return digits


"""
Calculates the decimal value of a number given as 0.[digits] * base^exponent
"""
def get_decimal_from_normalized_digits(digits, sign, base, exponent):
    result = Decimal(0)
    exponent -= 1
    for i in range(len(digits)):
        result += digits[i] * (base ** exponent)
        exponent -= 1

    return sign * result


"""
Calculates the first <digitCount> digits (unrounded) of the given number in the specified base, on the
normalized format 0.XXXXX * base ^ exponent

Returns: array of digits and exponent.
"""
def convert_to_base_normalized(x, b, digitCount):
    digits = []
    exponent = Decimal(0)
    x_dec_abs = Decimal(abs(x))
    base = Decimal(b)

    # find exponent
    if x_dec_abs > base:
        while base ** exponent <= x_dec_abs:
            exponent += 1
    elif x_dec_abs < base:
        while base ** exponent >= x_dec_abs:
            exponent -= 1
        exponent += 1

    # calculate digits
    current_exponent = exponent - 1
    for i in range(digitCount):
        digit = math.floor(x_dec_abs / (base ** current_exponent))
        digits.append(digit)
        x_dec_abs = x_dec_abs - (digit * (base ** current_exponent))
        current_exponent -= 1

    return digits, exponent


"""
Prints a number given by its normalized digits, exponent and base; base between 2 and 16 inclusive.
"""
def print_normalized_number(digits, sign, base, exponent):
    digit_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    sign_name = "" if sign == 1 else "-"
    return sign_name + "(0." + "".join([digit_names[d] for d in digits]) + ")_" + str(base) + " * " + str(base) + "^" + str(exponent)


def convert_to_machine_number(x, b, digitCount):
    # convert to base
    double_digits, exponent = convert_to_base_normalized(x, b, digitCount * 2)

    digits = double_digits.copy()[0:math.floor((len(double_digits) + 1) / 2)]
    x_dec_abs = Decimal(abs(x))
    sign = 1 if x >= 0 else -1

    print("Zahl ins gewünschte Zahlensystem umrechnen und normalisieren:")
    print("x = " + str(x) + " = " + print_normalized_number(double_digits, sign, b, exponent) + "\n")

    # round correctly
    digits_rounded_up = round_up(digits.copy(), b)

    # if all digits overflow, there's one more digit --> exponent + 1
    exponent_rounded_up = exponent if len(digits) == len(digits_rounded_up) else exponent + 1

    rounded_down = get_decimal_from_normalized_digits(digits, sign, Decimal(b), exponent)
    rounded_up = get_decimal_from_normalized_digits(digits_rounded_up, sign, Decimal(b), exponent_rounded_up)

    if abs(x_dec_abs - rounded_down) > abs(x_dec_abs - rounded_up):
        digits = digits_rounded_up[0:digitCount]
        exponent = exponent_rounded_up

    print("Auf " + str(digitCount) + " Nachkommastellen runden:")
    print("x~ = " + print_normalized_number(digits, sign, b, exponent) + "\n")

    return digits, sign, exponent


"""
MAIN PROGRAM
"""
digits, sign, exponent = convert_to_machine_number(x, base, digitCount)
value = get_decimal_from_normalized_digits(digits, sign, base, exponent)
relative_error = abs(Decimal(x) - value) / abs(Decimal(x))
absolute_error = abs(Decimal(x) - value)

print("Genauer Wert (Dezimal) x = " + str(Decimal(x)))
print("Wert der Maschinenzahl (Dezimal) x~ = " + str(value))
print("Relativer Fehler = |x - x~| / |x| = " + str(relative_error))
print("Absoluter Fehler = |x - x~| = " + str(absolute_error))
