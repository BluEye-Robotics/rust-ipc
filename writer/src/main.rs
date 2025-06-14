use ipc_lib::IPC;

#[repr(C)]
#[derive(Debug, Default)]
struct Data {
    counter: u32,
    value: f32,
}

fn main() {
    let shm = IPC::<Data>::new("/my_topic").unwrap();
    let data = shm.get();

    data.counter += 1;
    data.value = 3.14;
    println!("Wrote: {:?}", data);
}
