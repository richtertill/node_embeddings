Since a large number of people can contribute to this repository, we should 
stick to some conventions to make sure that the code is easy to read and doesn't
become messy.

### Git Commit Messages
* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Style Guide
* Use the general Python style recommendations from [PEP-8](https://www.python.org/dev/peps/pep-0008/).
* Use type hints (annotations)
* Use the [numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) (without types).
* Use `str.format` instead of f-strings to enable support for Python 3.5 and earlier.
* Adhere to a 120 character linewidth for code and 80 characters for docstrings.

### Adding new features
If you want to add a new feature to the package, you should do the following:
1. Create a new feature branch.
2. Add the new functionality there and commit it.
3. Create a merge request and ask someone to review your code.
4. Implement and / or discuss the suggestions of the reviewer, such that both of
you are happy with the result.
5. Merge your feature branch into the master branch.