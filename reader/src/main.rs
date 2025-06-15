use ipc_lib::IPC;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct Data {
    counter: u8,
    value: f32,
}

fn main() {
    match IPC::<Data>::new("/my/topic") {
        Ok(mut shm) => {
            let mut data = Data::default();
            let success = shm.get(&mut data);
            println!("Read Succes {:?}: {:?}", success, data);
        }
        Err(e) => {
            eprintln!("Failed to read shared memory: {}", e);
        }
    }
}
