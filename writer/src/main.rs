use ipc_lib::IPC;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct Data {
    counter: u32,
    value: f32,
}

fn main() {
    let shm = IPC::<Data>::new("/my_topic").unwrap();
    let data = Data {
        counter: 3,
        value: 3.14,
    };

    shm.set(data);
    println!("Wrote: {:?}", data);
}
