import * as THREE from 'three';
import { ComicPops } from '../ui/ComicPops';
import { sounds } from '../audio/SoundEffects';

export interface JackpotBeamOptions {
  position: THREE.Vector3;
  durationSeconds?: number;
  beamColor?: number;
  camera?: THREE.Camera;
}

/**
 * JackpotBeam (Section 15, Section 66 & Section 77).
 * Spawns an animated vertical golden light beam shooting into the sky,
 * dual rotating red emergency beacon warning lights, and emergency siren alarm audio.
 */
export class JackpotBeam {
  public mesh: THREE.Group;
  public position: THREE.Vector3;
  public isAlive: boolean = true;

  private duration: number;
  private elapsed: number = 0;
  private beamMesh: THREE.Mesh;
  private coreMesh: THREE.Mesh;
  private beaconGroup: THREE.Group;
  private helixRing: THREE.Mesh;

  private beamMat: THREE.MeshBasicMaterial;
  private coreMat: THREE.MeshBasicMaterial;
  private sirenAudioTimer: number = 0;

  constructor(scene: THREE.Scene, opts: JackpotBeamOptions) {
    this.position = opts.position.clone();
    this.duration = opts.durationSeconds || 10.0;

    this.mesh = new THREE.Group();
    this.mesh.position.copy(this.position);

    const beamColor = opts.beamColor || 0xFFD700;

    // 1. Outer Golden Volumetric Beam (Cylinder h=24m, r=0.65m)
    const beamGeo = new THREE.CylinderGeometry(0.5, 0.75, 24, 16, 1, true);
    beamGeo.translate(0, 12, 0);
    this.beamMat = new THREE.MeshBasicMaterial({
      color: beamColor,
      transparent: true,
      opacity: 0.55,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.beamMesh = new THREE.Mesh(beamGeo, this.beamMat);
    this.mesh.add(this.beamMesh);

    // 2. Inner Intense Core Beam (Cylinder h=24m, r=0.22m)
    const coreGeo = new THREE.CylinderGeometry(0.18, 0.28, 24, 12, 1, true);
    coreGeo.translate(0, 12, 0);
    this.coreMat = new THREE.MeshBasicMaterial({
      color: 0xFFFFFF,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.coreMesh = new THREE.Mesh(coreGeo, this.coreMat);
    this.mesh.add(this.coreMesh);

    // 3. Orbiting Gold Helix Ring
    const ringGeo = new THREE.TorusGeometry(0.9, 0.06, 8, 24);
    ringGeo.rotateX(Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xFFE066,
      transparent: true,
      opacity: 0.8,
    });
    this.helixRing = new THREE.Mesh(ringGeo, ringMat);
    this.helixRing.position.y = 1.0;
    this.mesh.add(this.helixRing);

    // 4. Dual Rotating Red Emergency Siren Beacon
    this.beaconGroup = new THREE.Group();
    this.beaconGroup.position.y = 1.2;
    const beaconGeo = new THREE.ConeGeometry(0.35, 1.2, 8);
    beaconGeo.rotateX(Math.PI / 2);
    const beaconMat = new THREE.MeshBasicMaterial({
      color: 0xFF1A1A,
      transparent: true,
      opacity: 0.75,
      side: THREE.DoubleSide,
    });
    const beacon1 = new THREE.Mesh(beaconGeo, beaconMat);
    beacon1.position.z = 0.5;
    const beacon2 = new THREE.Mesh(beaconGeo, beaconMat);
    beacon2.position.z = -0.5;
    beacon2.rotation.y = Math.PI;
    this.beaconGroup.add(beacon1);
    this.beaconGroup.add(beacon2);
    this.mesh.add(this.beaconGroup);

    scene.add(this.mesh);

    // Initial siren audio & comic pop
    sounds.playHonk();
    if (opts.camera) {
      ComicPops.spawnWorld(
        new THREE.Vector3(this.position.x, this.position.y + 2.5, this.position.z),
        opts.camera,
        'JACKPOT! 🚨',
        '#FFD700'
      );
    }
  }

  public update(delta: number, camera?: THREE.Camera) {
    if (!this.isAlive) return;
    this.elapsed += delta;

    // Rotate beams & beacon lights
    this.beamMesh.rotation.y += delta * 1.5;
    this.coreMesh.rotation.y -= delta * 2.2;
    this.beaconGroup.rotation.y += delta * 8.0; // High-speed rotating siren

    // Orbiting helix ring ascends
    this.helixRing.position.y = 1.0 + ((this.elapsed * 4.0) % 20.0);
    this.helixRing.rotation.z += delta * 3.0;

    // Siren alarm audio pulses periodically every 2.5s
    this.sirenAudioTimer += delta;
    if (this.sirenAudioTimer >= 2.5) {
      this.sirenAudioTimer = 0;
      sounds.playHonk();
      if (camera && Math.random() < 0.5) {
        ComicPops.spawnWorld(
          new THREE.Vector3(this.position.x, this.position.y + 2.2, this.position.z),
          camera,
          'WEE-WOO! 🚨',
          '#FF3366'
        );
      }
    }

    // Fade out near end of duration
    const remaining = this.duration - this.elapsed;
    if (remaining <= 2.0) {
      const alpha = Math.max(0, remaining / 2.0);
      this.beamMat.opacity = 0.55 * alpha;
      this.coreMat.opacity = 0.85 * alpha;
    }

    if (this.elapsed >= this.duration) {
      this.isAlive = false;
    }
  }

  public destroy(scene: THREE.Scene) {
    scene.remove(this.mesh);
  }
}

/**
 * JackpotFXManager:
 * Coordinates all active jackpot light beams and sirens in the scene.
 */
export class JackpotFXManager {
  private activeBeams: JackpotBeam[] = [];
  private scene: THREE.Scene;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  public spawnJackpot(opts: JackpotBeamOptions): JackpotBeam {
    const beam = new JackpotBeam(this.scene, opts);
    this.activeBeams.push(beam);
    return beam;
  }

  public update(delta: number, camera?: THREE.Camera) {
    for (let i = this.activeBeams.length - 1; i >= 0; i--) {
      const beam = this.activeBeams[i];
      beam.update(delta, camera);

      if (!beam.isAlive) {
        beam.destroy(this.scene);
        this.activeBeams.splice(i, 1);
      }
    }
  }

  public getActiveCount(): number {
    return this.activeBeams.length;
  }

  public clear() {
    for (const beam of this.activeBeams) {
      beam.destroy(this.scene);
    }
    this.activeBeams = [];
  }
}
