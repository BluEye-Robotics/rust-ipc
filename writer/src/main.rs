use ipc_lib::IPC;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct Data {
    counter: u8,
    value: f32,
}

fn main() {
    env_logger::init();
    let shm = IPC::<Data>::new("/my/topic").unwrap();

    let data = Data {
        counter: 3,
        value: 3.14,
    };

    shm.set(data);
    println!("Wrote: {:?}", data);
}
