import ast


def check_function_in_file():
    with open("task_1.py", "r") as file:
        file_content = file.read()

    tree = ast.parse(file_content)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "diagonals_sum":
                for arg in node.args.args:
                    if arg.arg == "matrix":
                        return True
    return False


def main():
    return check_function_in_file()


if __name__ == "__main__":
    print(main())
