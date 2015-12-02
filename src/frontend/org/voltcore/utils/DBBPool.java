/* This file is part of VoltDB.
 * Copyright (C) 2008-2015 VoltDB Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.voltcore.utils;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;
import com.google_voltpatches.common.base.Preconditions;
import org.cliffc_voltpatches.high_scale_lib.NonBlockingHashMap;
import org.voltcore.logging.VoltLogger;
import sun.misc.Cleaner;
import sun.nio.ch.DirectBuffer;
import java.util.List;
import java.util.ArrayList;
/**
 * A pool of {@link java.nio.ByteBuffer ByteBuffers} that are
 * allocated with
 * {@link java.nio.ByteBuffer#allocateDirect(int) * ByteBuffer.allocateDirect}.
 * Buffers are stored in Arenas that are powers of 2. The smallest arena is 16 bytes.
 * Arenas will shrink every 60 seconds if some of the memory isn't being used.
 */
public final class DBBPool {
    private static final VoltLogger TRACE = new VoltLogger("DBBPOOL");
    private static final VoltLogger HOST = new VoltLogger("DBBPOOL");
    static {
        final String msg = "Strict java memory checking is enabled, don't do release builds " +
                     "or performance runs with this enabled. Invoke \"ant clean\" and \"ant -Djmemcheck=NO_MEMCHECK\" to disable.";
        HOST.warn(msg);
        System.err.println(msg);
    }
    /**
     * Abstract base class for a ByteBuffer container. A container serves to hold a reference
     * to the pool/arena/whatever the ByteBuffer was allocated from and possibly the address
     * of the ByteBuffer if it is a DirectByteBuffer. The container also provides the interface
     * for discarding the ByteBuffer and returning it back to the pool. It is a good practice
     * to discard a container even if it is wrapper for HeapByteBuffer that isn't pooled.
     *
     */
    public static abstract class BBContainer {
        /**
         * The buffer
         */
        final private ByteBuffer b;
        private volatile Throwable m_freeThrowable;
        private final Throwable m_allocationThrowable;
        private List<String> m_tags = null;
        public BBContainer(ByteBuffer b) {
            m_allocationThrowable = new Throwable("\"" + Thread.currentThread().getName() + "\" at " + System.currentTimeMillis());
            this.b = b;
        }
        final public long address() {
            checkUseAfterFree();
            return ((DirectBuffer)b).address();
        }
        public void discard() {
            checkDoubleFree();
        }
        public ByteBuffer b() {
            checkUseAfterFree();
            return b;
        }
        final public ByteBuffer bD() {
            return b().duplicate();
        }
        final public ByteBuffer bDR() {
            return b().asReadOnlyBuffer();
        }
        final protected void checkUseAfterFree() {
            if (m_freeThrowable != null) {
                System.err.println("Use after free in DBBPool");
                System.err.println("Free was by:");
                m_freeThrowable.printStackTrace();
                System.err.println("Use was by:");
                Throwable t = new Throwable("\"" + Thread.currentThread().getName() + "\" at " + System.currentTimeMillis());
                t.printStackTrace();
                if (isTagged()) {
                    for (String tag: m_tags) {
                        System.err.println(tag);
                    }
                }
                HOST.fatal("Use after free in DBBPool");
                HOST.fatal("Free was by:", m_freeThrowable);
                HOST.fatal("Use was by:", t);
                System.exit(-1);
            }
        }
        final protected ByteBuffer checkDoubleFree() {
            synchronized (this) {
                if (m_freeThrowable != null) {
                    System.err.println("Double free in DBBPool");
                    System.err.println("Original free was by:");
                    m_freeThrowable.printStackTrace();
                    System.err.println("Current free was by:");
                    Throwable t = new Throwable("\"" + Thread.currentThread().getName() + "\" at " + System.currentTimeMillis());
                    t.printStackTrace();
                    if (isTagged()) {
                        for (String tag: m_tags) {
                            System.err.println(tag);
                        }
                    }
                    HOST.fatal("Double free in DBBPool");
                    HOST.fatal("Original free was by:", m_freeThrowable);
                    HOST.fatal("Current free was by:", t);
                    System.exit(-1);
                }
                m_freeThrowable = new Throwable("\"" + Thread.currentThread().getName() + "\" at " + System.currentTimeMillis());
            }
            return b;
        }
        public final void tag(final String tag) {
            StringBuilder sb = new StringBuilder(1024);
            sb.append("<<TAG:").append(m_tags.size()).append(">> ");
            sb.append(tag).append("\n");
            sb.append(CoreUtils.throwableToString(new Throwable()));
            synchronized(this) {
                if (m_tags == null) {
                    m_tags = new ArrayList<String>();
                }
                m_tags.add(sb.toString());
            }
        }
        public final void addToTagTrail(final String tag) {
            if (isTagged()) tag(tag);
        }
        public boolean isTagged() {
            return m_tags != null && !m_tags.isEmpty();
        }
        @Override
        public void finalize() {
            if (m_freeThrowable == null) {
                System.err.println("BBContainer " + Integer.toHexString(this.hashCode()) + " was never discarded allocated by:");
                m_allocationThrowable.printStackTrace();
                if (isTagged()) {
                    for (String tag: m_tags) {
                        System.err.println(tag);
                    }
                }
                HOST.fatal("BBContainer " + Integer.toHexString(this.hashCode()) + " was never discarded allocated by:", m_allocationThrowable);
                System.exit(-1);
            }
        }
    }
    /**
     * Wrapper for HeapByteBuffers that allows them to pose as ByteBuffers from a pool.
     *
     */
    public static final class BBWrapperContainer extends BBContainer {
        private BBWrapperContainer(ByteBuffer b) {
            super( b );
        }
        @Override
        public final void discard() {
            super.discard();
        }
    }
    public static final class DBBWrapperContainer extends BBContainer {
        private DBBWrapperContainer(ByteBuffer b) {
            super( b );
        }
        @Override
        public final void discard() {
            final ByteBuffer buf = checkDoubleFree();
            DBBPool.cleanByteBuffer(buf);
        }
    }
    public static class MBBContainer extends BBContainer {
        private MBBContainer(MappedByteBuffer buf) {
            super(buf);
        }
        @Override
        public MappedByteBuffer b() {
            return (MappedByteBuffer)super.b();
        }
        @Override
        public void discard() {
            final ByteBuffer buf = checkDoubleFree();
            DBBPool.cleanByteBuffer(buf);
        }
    }
    public static final BBContainer dummyWrapBB(ByteBuffer b) {
        return new BBWrapperContainer(b);
    }
    /**
     * Static factory method to wrap a ByteBuffer in a BBContainer that is not
     * associated with any pool
     * @param b
     */
    public static final BBContainer wrapBB(ByteBuffer b) {
        if (b.isDirect()) {
            return new DBBWrapperContainer(b);
        } else {
            return new BBWrapperContainer(b);
        }
    }
    public static final MBBContainer wrapMBB(ByteBuffer b) {
        Preconditions.checkArgument(b.isDirect());
        return new MBBContainer((MappedByteBuffer)b);
    }
    /**
     * Number of bytes allocated globally by DBBPools
     */
    private static AtomicLong bytesAllocatedGlobally = new AtomicLong(0);
    static long getBytesAllocatedGlobally()
    {
        return bytesAllocatedGlobally.get();
    }
    private static final VoltLogger m_logger = new VoltLogger(DBBPool.class.getName());
    /**
     * Retrieve the CRC32C value of a DirectByteBuffer as a long
     * The polynomial is different from java.util.zip.CRC32C,
     * and matches the one used by SSE 4.2. hardware CRC instructions.
     * The implementation will use the SSE 4.2. instruction if the native library
     * was compiled with -msse4.2 and there is hardware support, otherwise it falls
     * back to Intel's slicing by 8 algorithm
     * @param b Buffer you want to retrieve the CRC32 of
     * @param offset Offset into buffer to start calculations
     * @param length Length of the buffer to calculate
     * @return CRC32C of the buffer as an int.
     */
    public static native int getBufferCRC32C( ByteBuffer b, int offset, int length);
    /**
     * Retrieve the CRC32C value of a DirectByteBuffer as a long
     * The polynomial is different from java.util.zip.CRC32C,
     * and matches the one used by SSE 4.2. hardware CRC instructions.
     * The implementation will use the SSE 4.2. instruction if the native library
     * was compiled with -msse4.2 and there is hardware support, otherwise it falls
     * back to Intel's slicing by 8 algorithm
     * @param ptr Address of buffer you want to retrieve the CRC32C of
     * @param offset Offset into buffer to start calculations
     * @param length Length of the buffer to calculate
     * @return CRC32C of the buffer as an int.
     */
    public static native int getCRC32C( long ptr, int offset, int length);
    /**
     * Retrieve the CRC32 value of a DirectByteBuffer as a long
     * @param b Buffer you want to retrieve the CRC32 of
     * @param offset Offset into buffer to start calculations
     * @param length Length of the buffer to calculate
     * @return CRC32 of the buffer as an int.
     */
    public static native int getBufferCRC32( ByteBuffer b, int offset, int length);
    /**
     * Retrieve the CRC32 value of a DirectByteBuffer as a long
     * @param ptr Address of buffer you want to retrieve the CRC32 of
     * @param offset Offset into buffer to start calculations
     * @param length Length of the buffer to calculate
     * @return CRC32 of the buffer as an int.
     */
    public static native int getCRC32( long ptr, int offset, int length);
    /**
     * Retrieve the first 8 bytes of the Murmur hash3_x64_128 of DirectByteBuffer a
     * as a long
     * @param ptr pointer to the buffer
     * @param offset Offset into buffer to start calculations
     * @param length Length of the buffer to calculate
     * @return First 8 bytes of  Murmur hash3_x64_128 of buffer
     */
    public static native int getMurmur3128( long ptr, int offset, int length);
    /**
     * Retrieve the first 8 bytes of the Murmur hash3_x64_128 of long value
     * @param value value to hash
     * @return First 8 bytes of  Murmur hash3_x64_128 of value
     */
    public static native int getMurmur3128( long value);
    private static final NonBlockingHashMap<Integer, ConcurrentLinkedQueue<BBContainer>> m_pooledBuffers =
            new NonBlockingHashMap<Integer, ConcurrentLinkedQueue<BBContainer>>();
    /*
     * Allocate a DirectByteBuffer from a global lock free pool
     */
    public static BBContainer allocateDirectAndPool(final Integer capacity) {
        ConcurrentLinkedQueue<BBContainer> pooledBuffers = m_pooledBuffers.get(capacity);
        if (pooledBuffers == null) {
            pooledBuffers = new ConcurrentLinkedQueue<BBContainer>();
            if (m_pooledBuffers.putIfAbsent(capacity, pooledBuffers) == null) {
                pooledBuffers = m_pooledBuffers.get(capacity);
            }
        }
        BBContainer cont = pooledBuffers.poll();
        if (cont == null) {
            cont = allocateDirect(capacity);
        }
        final BBContainer origin = cont;
        cont = new BBContainer(origin.b()) {
            @Override
            public void discard() {
                final ByteBuffer b = checkDoubleFree();
                m_pooledBuffers.get(b.capacity()).offer(origin);
            }
        };
        cont.b().clear();
        return cont;
    }
    //In OOM conditions try clearing the pool
    private static void clear() {
        long startingBytes = bytesAllocatedGlobally.get();
        for (ConcurrentLinkedQueue<BBContainer> pool : m_pooledBuffers.values()) {
            BBContainer cont = null;
            while ((cont = pool.poll()) != null) {
                cont.discard();
            }
        }
        new VoltLogger("HOST").warn(
                "Attempted to resolve DirectByteBuffer OOM by freeing pooled buffers. " +
                "Starting bytes was " + startingBytes + " after clearing " +
                 bytesAllocatedGlobally.get() + " change " + (startingBytes - bytesAllocatedGlobally.get()));
    }
    private static void logAllocation(int capacity) {
        if (TRACE.isTraceEnabled()) {
            String message =
                    "Allocated DBB capacity " + capacity +
                     " total allocated " + bytesAllocatedGlobally.get() +
                     " from " + CoreUtils.throwableToString(new Throwable());
            TRACE.trace(message);
        }
    }
    private static void logDeallocation(int capacity) {
        if (TRACE.isTraceEnabled()) {
            String message =
                    "Deallocated DBB capacity " + capacity +
                    " total allocated " + bytesAllocatedGlobally.get() +
                    " from " + CoreUtils.throwableToString(new Throwable());
            TRACE.trace(message);
        }
    }
    /*
     * The only reason to not retrieve the address is that network code shared
     * with the java client shouldn't have a dependency on the native library
     */
    public static BBContainer allocateDirect(final int capacity) {
        ByteBuffer retval = null;
        try {
            retval = ByteBuffer.allocateDirect(capacity);
        } catch (OutOfMemoryError e) {
            if (e.getMessage().contains("Direct buffer memory")) {
                clear();
                retval = ByteBuffer.allocateDirect(capacity);
            } else {
                throw new Error(e);
            }
        }
        bytesAllocatedGlobally.getAndAdd(capacity);
        logAllocation(capacity);
        return new DeallocatingContainer(retval);
    }
    private static class DeallocatingContainer extends BBContainer {
        private DeallocatingContainer(ByteBuffer buf) {
            super(buf);
        }
        @Override
        public void discard() {
            final ByteBuffer buf = checkDoubleFree();
            try {
                bytesAllocatedGlobally.getAndAdd(-buf.capacity());
                logDeallocation(buf.capacity());
                DBBPool.cleanByteBuffer(buf);
            } catch (Throwable e) {
                // The client code doesn't want to link to the VoltDB class, so this hack was born.
                // It should be temporary as the goal is to remove client code dependency on
                // DBBPool in the medium term.
                try {
                    Class<?> vdbClz = Class.forName("org.voltdb.VoltDB");
                    Method m = vdbClz.getMethod("crashLocalVoltDB", String.class, boolean.class, Throwable.class);
                    m.invoke(null, "Failed to deallocate direct byte buffer", false, e);
                } catch (Exception ignored) {
                    System.err.println("Failed to deallocate direct byte buffer");
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }
    public static void registerUnsafeMemory(long pointer) {
    }
    /*
     * Delete a char array that was allocated on the native heap
     *
     * If you want to debug issues with double free, uncomment m_deletedStuff
     * and the code that populates it and comment out the call to nativeDeleteCharArrayMemory
     * and it will validate that nothing is ever deleted twice at the cost of unbounded memory usage
     */
    private static void deleteCharArrayMemory(long pointer) {
        nativeDeleteCharArrayMemory(pointer);
    }
    private static native void nativeDeleteCharArrayMemory(long pointer);
    public static BBContainer allocateUnsafeByteBuffer(long size) {
        final BBContainer retcont = DBBPool.wrapBB(nativeAllocateUnsafeByteBuffer(size));
        return retcont;
    }
    /*
     * Allocate a direct byte buffer that bypasses all GC and Java limits
     * and requires manual memory management. The memory will not be zeroed
     * and will come from the new/delete in C++. The pointer can be freed
     * using deleteCharArrayMemory.
     */
    private static native ByteBuffer nativeAllocateUnsafeByteBuffer(long size);
    /*
     * For managed buffers runs the cleaner, if there is no cleaner,
     * called deleteCharArrayMemory on the address
     */
    private static void cleanByteBuffer(ByteBuffer buf) {
        if (buf == null) return;
        if (!buf.isDirect()) return;
        final DirectBuffer dbuf = (DirectBuffer) buf;
        final Cleaner cleaner = dbuf.cleaner();
        if (cleaner != null) cleaner.clean();
        else deleteCharArrayMemory(dbuf.address());
    }
}
