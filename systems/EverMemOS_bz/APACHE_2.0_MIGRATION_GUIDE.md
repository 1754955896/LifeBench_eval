# Apache 2.0 License Migration Guide

## ✅ What Has Been Done

1. **Updated README files** - Both `README.md` and `README_zh.md` have been updated to:
   - Change license badge from MIT to Apache 2.0
   - Add language selection links (English | 简体中文)
   - Update license section with Apache 2.0 requirements

2. **Replaced LICENSE file** - The MIT License has been replaced with the full Apache License 2.0 text

3. **Created NOTICE file** - Added a NOTICE file as required by Apache 2.0

## 📋 What You Still Need to Do

### 1. Add License Headers to Source Files (Recommended)

Add the following header to the top of each significant source file (`.py`, `.js`, etc.):

```python
# Copyright 2025 EverMem AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**You can automate this using a script:**

```bash
# For Python files
find src -name "*.py" -type f -exec sed -i '1i\
# Copyright 2025 EverMem AI\n\
#\n\
# Licensed under the Apache License, Version 2.0 (the "License");\n\
# you may not use this file except in compliance with the License.\n\
# You may obtain a copy of the License at\n\
#\n\
#     http://www.apache.org/licenses/LICENSE-2.0\n\
#\n\
# Unless required by applicable law or agreed to in writing, software\n\
# distributed under the License is distributed on an "AS IS" BASIS,\n\
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n\
# See the License for the specific language governing permissions and\n\
# limitations under the License.\n' {} \;
```

### 2. Update pyproject.toml

Update the license field in your `pyproject.toml`:

```toml
[project]
license = {text = "Apache-2.0"}
# or
license = "Apache-2.0"
```

### 3. Review Third-Party Dependencies

Check if any of your dependencies have license conflicts with Apache 2.0:

```bash
# Install pip-licenses
pip install pip-licenses

# Check licenses
pip-licenses --format=markdown --output-file=THIRD_PARTY_LICENSES.md
```

**Important**: Some licenses (like GPL) may not be compatible with Apache 2.0. Review and ensure compatibility.

### 4. Update NOTICE File (If Needed)

If you use any third-party Apache-licensed code, add attribution to the NOTICE file:

```
This product includes software developed by [Project Name] ([URL]).
```

### 5. Create CONTRIBUTING.md (If Not Exists)

Add a section about license compliance:

```markdown
## License Agreement

By contributing to this project, you agree that your contributions will be licensed under the Apache License 2.0.
```

### 6. Notify Existing Contributors (Important!)

Since you're changing from MIT to Apache 2.0, it's good practice to:

1. **Post an announcement** in your repository (GitHub Discussions or Issues)
2. **Notify all previous contributors** about the license change
3. **Get explicit consent** if possible (especially if there have been significant contributions)

Example announcement:

```markdown
## License Change: MIT → Apache 2.0

We're migrating from MIT License to Apache License 2.0 to provide better patent protection 
and clearer terms for commercial use.

**What this means:**
- Users must include the Apache 2.0 license text
- Users must state changes made to the code
- Better patent protection for users and contributors

If you have contributed to this project and have concerns about this change, please comment below.
```

### 7. Update CI/CD (If Applicable)

Add license checking to your CI pipeline:

```yaml
# Example for GitHub Actions
- name: Check License Headers
  run: |
    # Add script to verify all source files have proper headers
```

### 8. Update Documentation

Check and update any other documentation that mentions the license:
- API documentation
- Developer guides
- Website (if any)
- Package metadata

## 🔍 Key Differences: MIT vs Apache 2.0

| Aspect | MIT | Apache 2.0 |
|--------|-----|------------|
| **Patent Grant** | No explicit patent grant | Explicit patent rights grant |
| **Trademark** | Not addressed | Explicitly does not grant trademark rights |
| **Attribution** | Simple copyright notice | Requires keeping NOTICE file and attributions |
| **Changes** | No requirement to state changes | Must state significant changes made |
| **Length** | Very short (~200 words) | Much longer (full legal document) |
| **Compatibility** | Highly compatible | Some restrictions (not GPL compatible) |

## ✨ Benefits of Apache 2.0

1. **Patent Protection** - Explicit patent license protects users from patent claims
2. **Contributor Protection** - Clear terms for contributions
3. **Enterprise Friendly** - Preferred by many enterprises for commercial use
4. **Trademark Protection** - Protects your project name and branding
5. **Change Tracking** - Requires documentation of modifications

## ⚠️ Legal Disclaimer

**I am not a lawyer.** This guide provides general information about Apache 2.0 migration. 

For legal advice specific to your situation, especially if:
- You have significant external contributors
- Your project is already widely used
- You're concerned about legal implications

Please consult with a qualified attorney.

## 📚 Additional Resources

- [Apache License 2.0 Official Text](https://www.apache.org/licenses/LICENSE-2.0)
- [Apache License FAQ](https://www.apache.org/foundation/license-faq.html)
- [Choosing a License Guide](https://choosealicense.com/)
- [GitHub License Help](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

## ✅ Checklist

- [x] Updated LICENSE file
- [x] Updated README files
- [x] Created NOTICE file
- [ ] Add license headers to source files
- [ ] Update pyproject.toml
- [ ] Review third-party dependencies
- [ ] Update NOTICE file with third-party attributions
- [ ] Notify existing contributors
- [ ] Update CONTRIBUTING.md
- [ ] Update CI/CD pipeline
- [ ] Update all documentation
- [ ] Make announcement about license change

---

**Next Steps**: Review this checklist and complete the remaining items. Start with adding license headers to source files and updating pyproject.toml.

