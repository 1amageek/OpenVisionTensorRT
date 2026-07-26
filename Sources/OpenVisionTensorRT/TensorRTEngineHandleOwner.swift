import CTensorRTShim
import Synchronization

final class TensorRTEngineHandleOwner: Sendable {
    // The address remains bound to OVTRTEngine until consume clears it.
    // Swift never dereferences, offsets, or exports it. One Mutex protects the
    // same storage and exactly-once destruction contract on every target.
    private let address: Mutex<UInt?>

    init(handle: OpaquePointer) {
        address = Mutex(UInt(bitPattern: handle))
    }

    deinit {
        if let handle = consume() {
            ovtrt_engine_destroy(handle)
        }
    }

    var isActive: Bool {
        address.withLock { $0 != nil }
    }

    func withHandle<Result>(
        _ body: (OpaquePointer) throws -> Result
    ) rethrows -> Result? {
        let bits = address.withLock { $0 }
        guard
            let bits,
            let handle = OpaquePointer(bitPattern: bits)
        else {
            return nil
        }
        return try body(handle)
    }

    func consume() -> OpaquePointer? {
        let bits = address.withLock { address in
            let bits = address
            address = nil
            return bits
        }
        guard let bits else {
            return nil
        }
        return OpaquePointer(bitPattern: bits)
    }
}
