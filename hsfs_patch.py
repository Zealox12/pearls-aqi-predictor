# hsfs_patch.py - Import this BEFORE importing hsfs
import sys

# Mock pyjks
sys.modules['pyjks'] = type(sys)('pyjks')
sys.modules['pyjks'].KeyStore = None

# Mock great_expectations
import types
ge = types.ModuleType('great_expectations')
ge.core = types.ModuleType('great_expectations.core')
ge.core.expectation_validation_result = types.ModuleType('great_expectations.core.expectation_validation_result')

class MockValidationResult:
    def __init__(self, *args, **kwargs):
        pass

ge.core.expectation_validation_result.ExpectationValidationResult = MockValidationResult
sys.modules['great_expectations'] = ge
sys.modules['great_expectations.core'] = ge.core
sys.modules['great_expectations.core.expectation_validation_result'] = ge.core.expectation_validation_result

# Mock altair (also imported by some hsfs paths)
altair = types.ModuleType('altair')
sys.modules['altair'] = altair