import Foundation

/// Pause / resume / stop control for one sync run. The requesting side (typically a UI) calls
/// the synchronous state transitions; the orchestrator suspends before mutations while paused,
/// observes stop during source collection, and checks both states between payloads.
///
/// Lock-based rather than an actor so callers can flip state synchronously — a pause must be
/// observable by the loop's very next checkpoint without an actor-hop race.
public final class SyncControl: @unchecked Sendable {
    public enum State: Sendable {
        case running, paused, stopping
    }

    private let lock = NSLock()
    private var _state: State = .running
    private var pauseWaiters: [UUID: CheckedContinuation<Void, Never>] = [:]
    private var stopWaiters: [UUID: CheckedContinuation<Void, Never>] = [:]

    public init() {}

    public var state: State {
        lock.withLock { _state }
    }

    public func pause() {
        lock.withLock {
            if _state == .running { _state = .paused }
        }
    }

    public func resume() {
        let resumed: [CheckedContinuation<Void, Never>] = lock.withLock {
            guard _state == .paused else { return [] }
            _state = .running
            defer { pauseWaiters.removeAll() }
            return Array(pauseWaiters.values)
        }
        for continuation in resumed { continuation.resume() }
    }

    public func stop() {
        let resumed: (
            pause: [CheckedContinuation<Void, Never>],
            stop: [CheckedContinuation<Void, Never>]
        ) = lock.withLock {
            guard _state != .stopping else { return ([], []) }
            _state = .stopping
            defer {
                pauseWaiters.removeAll()
                stopWaiters.removeAll()
            }
            return (Array(pauseWaiters.values), Array(stopWaiters.values))
        }
        for continuation in resumed.pause { continuation.resume() }
        for continuation in resumed.stop { continuation.resume() }
    }

    var isStopping: Bool {
        state == .stopping
    }

    func waitWhilePaused() async {
        while true {
            guard !Task.isCancelled else { return }
            guard lock.withLock({ _state == .paused }) else { return }
            let waiterID = UUID()
            await withTaskCancellationHandler {
                await withCheckedContinuation { continuation in
                    let resumeImmediately: Bool = lock.withLock {
                        guard _state == .paused, !Task.isCancelled else { return true }
                        pauseWaiters[waiterID] = continuation
                        return false
                    }
                    if resumeImmediately { continuation.resume() }
                }
            } onCancel: {
                let continuation = lock.withLock {
                    pauseWaiters.removeValue(forKey: waiterID)
                }
                continuation?.resume()
            }
        }
    }

    /// Suspend until stop is requested. Cancellation removes and resumes the caller's waiter,
    /// allowing an orchestrator to race this signal against connector collection safely.
    func waitUntilStopping() async {
        guard !Task.isCancelled else { return }
        guard lock.withLock({ _state != .stopping }) else { return }

        let waiterID = UUID()
        await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                let resumeImmediately: Bool = lock.withLock {
                    guard _state != .stopping, !Task.isCancelled else { return true }
                    stopWaiters[waiterID] = continuation
                    return false
                }
                if resumeImmediately { continuation.resume() }
            }
        } onCancel: {
            let continuation = lock.withLock {
                stopWaiters.removeValue(forKey: waiterID)
            }
            continuation?.resume()
        }
    }
}
