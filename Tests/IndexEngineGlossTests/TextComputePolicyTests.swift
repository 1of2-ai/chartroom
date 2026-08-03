import Foundation
import Testing
@testable import IndexEngineGloss

@Suite("TextComputePreference policy")
struct TextComputePolicyTests {
    @Test("the stored preference wins when Low Power Mode is off")
    func storedPreferenceWins() {
        #expect(TextComputePreference.effective(stored: .speed, isLowPowerModeEnabled: false) == .speed)
        #expect(TextComputePreference.effective(stored: .efficiency, isLowPowerModeEnabled: false) == .efficiency)
    }

    @Test("Low Power Mode hard-overrides to the Neural Engine")
    func lowPowerModeOverrides() {
        #expect(TextComputePreference.effective(stored: .speed, isLowPowerModeEnabled: true) == .efficiency)
    }

    @Test("a missing or unknown stored value falls back to efficiency")
    func missingStoredValueFallsBack() throws {
        #expect(TextComputePreference.effective(stored: nil, isLowPowerModeEnabled: false) == .efficiency)

        let suiteName = "TextComputePolicyTests-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defaults.removePersistentDomain(forName: suiteName)
        defer { defaults.removePersistentDomain(forName: suiteName) }

        defaults.set("not-a-mode", forKey: TextComputePreference.defaultsKey)
        // Live Low Power Mode state applies, but both branches resolve to a valid mode.
        let resolved = TextComputePreference.effective(defaults: defaults)
        #expect(resolved == .efficiency)
    }
}
