import CTensorRTShim
import Synchronization

final class TensorRTRuntimeHandleOwner: Sendable {
    // The C runtime pointer is created by ovtrt_runtime_create and remains
    // bound to OVTRTRuntime until exactly one call to ovtrt_runtime_destroy.
    // Swift never dereferences or offsets it. The integer representation is
    // protected by one Mutex on every target and is reconstructed only for the
    // C destruction call. `consume` clears the address before returning, so
    // explicit shutdown and deinitialization cannot both destroy the owner.
    private let address: Mutex<UInt?>

    init(handle: OpaquePointer) {
        address = Mutex(UInt(bitPattern: handle))
    }

    deinit {
        if let handle = consume() {
            ovtrt_runtime_destroy(handle)
        }
    }

    var isActive: Bool {
        address.withLock { $0 != nil }
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
