# 🤝 Contributing to ML Mastery Hub

We love your input! We want to make contributing to ML Mastery Hub as easy and transparent as possible, whether it's:

- 🐛 Reporting bugs
- 💡 Discussing the current state of the code
- 🚀 Submitting fixes
- 🎯 Proposing new features
- 📚 Adding tutorials or projects
- 🛠️ Improving tools and utilities

## 🚀 Quick Contribution Guide

### 1. Fork & Clone
```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/yourusername/ML-Mastery-Hub.git
cd ML-Mastery-Hub
```

### 2. Create a Branch
```bash
git checkout -b feature/amazing-new-feature
# or
git checkout -b bugfix/fix-important-bug
# or
git checkout -b tutorial/new-ml-concept
```

### 3. Make Changes
- Write your code
- Follow our coding standards
- Add tests if applicable
- Update documentation

### 4. Commit & Push
```bash
git add .
git commit -m "✨ Add amazing new feature"
git push origin feature/amazing-new-feature
```

### 5. Create Pull Request
- Go to GitHub and create a Pull Request
- Fill out the PR template
- Wait for review!

## 📋 Types of Contributions

### 🎪 Adding New Projects
Structure your project like this:
```
projects/[level]/[project-name]/
├── README.md          # Project documentation
├── main.py           # Main project file
├── requirements.txt  # Dependencies
├── data/            # Sample data (if applicable)
└── notebooks/       # Jupyter notebooks (if applicable)
```

**Project Requirements:**
- Clear difficulty level (⭐ ⭐⭐ ⭐⭐⭐)
- Comprehensive README
- Well-commented code
- Real-world dataset
- Learning objectives
- Expected outcomes

### 📖 Adding Tutorials
Structure your tutorial like this:
```
tutorials/[number]-[topic-name]/
├── README.md         # Tutorial overview
├── tutorial.ipynb   # Main notebook
├── images/          # Screenshots/diagrams
├── code/            # Supporting code files
└── exercises/       # Practice problems
```

**Tutorial Requirements:**
- Clear learning objectives
- Step-by-step explanations
- Visual aids and examples
- Practice exercises
- Links to additional resources

### 🔧 Adding Tools
Structure your tool like this:
```
tools/[tool-name]/
├── README.md         # Tool documentation
├── [tool_name].py   # Main tool file
├── tests/           # Unit tests
├── examples/        # Usage examples
└── requirements.txt # Dependencies
```

**Tool Requirements:**
- Clear documentation
- Usage examples
- Error handling
- Type hints
- Unit tests

## 📏 Coding Standards

### Python Code Style
- Follow **PEP 8** guidelines
- Use **Black** for formatting
- Add **type hints** where possible
- Include **docstrings** for all functions and classes

### Documentation
- Use **Markdown** for README files
- Include emojis for better readability 🎯
- Provide clear examples
- Link to relevant resources

### Commit Messages
Use **conventional commits** format:
```
type(scope): description

Examples:
✨ feat(projects): add stock prediction project
🐛 fix(tools): resolve AutoEDA memory issue
📚 docs(tutorials): update beginner guide
🎨 style(code): format with black
♻️ refactor(tools): improve model trainer performance
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests
- Place tests in `tests/` directories
- Name test files `test_*.py`
- Use descriptive test names
- Cover both happy and edge cases

Example test structure:
```python
def test_model_trainer_basic_functionality():
    """Test that ModelTrainer can train a simple model."""
    # Arrange
    trainer = ModelTrainer()
    X, y = load_test_data()
    
    # Act
    results = trainer.train(X, y)
    
    # Assert
    assert results['accuracy'] > 0.8
    assert 'best_model' in results
```

## 📝 Pull Request Process

### Before Submitting
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No merge conflicts

### PR Template
When creating a PR, include:

**Description**
- What does this PR do?
- Why is this change needed?

**Type of Change**
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📚 Documentation update
- [ ] 🎯 New project
- [ ] 🔧 New tool

**Testing**
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing completed

**Screenshots** (if applicable)

### Review Process
1. Automated checks must pass
2. At least one maintainer review
3. Address feedback promptly
4. Maintain civil discussion

## 🏷️ Issue Guidelines

### Bug Reports
Include:
- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, etc.
- **Screenshots**: If applicable

### Feature Requests
Include:
- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Other solutions considered
- **Additional Context**: Any other relevant info

### Tutorial/Project Requests
Include:
- **Topic**: What should be covered?
- **Difficulty**: Beginner/Intermediate/Advanced
- **Learning Objectives**: What will users learn?
- **Dataset**: Suggested dataset (if applicable)

## 🎯 Project Ideas We're Looking For

### Beginner Projects
- Simple classification/regression tasks
- Data visualization tutorials
- Basic feature engineering
- Model comparison studies

### Intermediate Projects
- Time series forecasting
- Natural language processing
- Computer vision basics
- Recommendation systems
- A/B testing analysis

### Advanced Projects
- Deep learning applications
- MLOps pipelines
- Custom model architectures
- Production deployment
- Advanced optimization

### Tools & Utilities
- Data preprocessing tools
- Model evaluation utilities
- Visualization helpers
- Deployment scripts
- Monitoring dashboards

## 🚨 Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience, nationality
- Personal appearance, race, religion
- Sexual identity and orientation

### Our Standards
**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other members

**Unacceptable behavior includes:**
- Trolling, insulting, or derogatory comments
- Public or private harassment
- Publishing others' private information
- Other conduct inappropriate in a professional setting

### Enforcement
Report unacceptable behavior to: ml.mastery.hub@gmail.com

## 🎖️ Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Social media shoutouts
- Special contributor badges

### Contributor Levels
- **🌱 Contributor**: Made at least 1 contribution
- **🌿 Regular Contributor**: Made 5+ contributions
- **🌳 Core Contributor**: Made 20+ contributions
- **🏆 Maintainer**: Trusted with review privileges

## 📚 Resources

### Learning Resources
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Markdown Guide](https://www.markdownguide.org/)
- [Python Style Guide](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### ML Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Papers with Code](https://paperswithcode.com/)

## ❓ Questions?

- 💬 Join our [GitHub Discussions](https://github.com/yourusername/ML-Mastery-Hub/discussions)
- 📧 Email us at: ml.mastery.hub@gmail.com
- 🐛 Create an [Issue](https://github.com/yourusername/ML-Mastery-Hub/issues)

---

**Thank you for contributing to ML Mastery Hub! 🚀**

Every contribution, no matter how small, makes a difference in helping others learn machine learning. Together, we're building something amazing! 🌟
