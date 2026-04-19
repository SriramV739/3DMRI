import React, { Suspense, useEffect, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Bounds, Center, Environment, Html, OrbitControls, useGLTF } from '@react-three/drei';
import { XR, createXRStore } from '@react-three/xr';
import { Glasses, RotateCcw } from 'lucide-react';
import * as THREE from 'three';

const xrStore = createXRStore();

function VolumeModel({ url, organFilterEnabled = false, visibleLabels = [] }) {
  const gltf = useGLTF(url);
  const visibleSet = useMemo(() => new Set(visibleLabels), [visibleLabels]);

  useEffect(() => {
    gltf.scene.traverse((object) => {
      if (!object.isMesh) return;
      object.frustumCulled = false;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      materials.forEach((material) => {
        if (!material) return;
        material.side = THREE.DoubleSide;
        material.vertexColors = true;
        if (material.opacity < 1 || material.transparent) {
          material.transparent = true;
          material.depthWrite = false;
          material.alphaTest = 0.02;
        }
        material.needsUpdate = true;
      });

      if (!organFilterEnabled) {
        object.visible = true;
        return;
      }

      const nameCandidates = [object.name, object.parent?.name, object.userData?.name]
        .filter((value) => typeof value === 'string' && value.trim().length > 0)
        .map((value) => value.trim());

      if (!nameCandidates.length) {
        object.visible = true;
        return;
      }

      object.visible = nameCandidates.some((name) => visibleSet.has(name));
    });
  }, [gltf, organFilterEnabled, visibleSet]);

  return (
    <Center>
      <primitive object={gltf.scene} />
    </Center>
  );
}

function EmptyState() {
  return (
    <Html center>
      <div className="scene-empty">Generate or select a CT volume</div>
    </Html>
  );
}

export default function VolumeScene({ volume, onReset, organFilterEnabled = false, visibleLabels = [] }) {
  const volumeUrl = useMemo(() => volume?.volume_url || '', [volume]);
  const [xrStatus, setXrStatus] = useState({
    checked: false,
    immersiveVR: false,
    message: 'Checking spatial mode',
  });

  useEffect(() => {
    let cancelled = false;

    async function detectXR() {
      if (!window.isSecureContext && !['localhost', '127.0.0.1'].includes(window.location.hostname)) {
        setXrStatus({
          checked: true,
          immersiveVR: false,
          message: 'Spatial mode needs HTTPS or localhost',
        });
        return;
      }

      if (!navigator.xr?.isSessionSupported) {
        setXrStatus({
          checked: true,
          immersiveVR: false,
          message: 'WebXR is unavailable in this browser',
        });
        return;
      }

      try {
        const immersiveVR = await navigator.xr.isSessionSupported('immersive-vr');
        if (!cancelled) {
          setXrStatus({
            checked: true,
            immersiveVR,
            message: immersiveVR ? 'Vision Pro spatial mode ready' : 'Enable WebXR Device API in Safari',
          });
        }
      } catch {
        if (!cancelled) {
          setXrStatus({
            checked: true,
            immersiveVR: false,
            message: 'WebXR support could not be verified',
          });
        }
      }
    }

    detectXR();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="spatial-pane">
      <div className="scene-toolbar">
        <button
          type="button"
          title={xrStatus.message}
          onClick={() => xrStore.enterVR()}
          disabled={!xrStatus.immersiveVR}
        >
          <Glasses size={18} />
          <span>Spatial</span>
        </button>
        <button type="button" title="Reset camera" onClick={onReset}>
          <RotateCcw size={18} />
        </button>
      </div>
      <Canvas camera={{ position: [0, 0.5, 5.2], fov: 48 }} gl={{ antialias: true, alpha: false }}>
        <color attach="background" args={['#071018']} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 5, 4]} intensity={1.8} />
        <directionalLight position={[-5, -2, -3]} intensity={0.7} color="#80d1ff" />
        <XR store={xrStore}>
          <Suspense fallback={<Html center><div className="scene-empty">Loading volume</div></Html>}>
            <Bounds fit clip observe margin={1.15}>
                {volumeUrl ? (
                  <VolumeModel
                    url={volumeUrl}
                    organFilterEnabled={organFilterEnabled}
                    visibleLabels={visibleLabels}
                  />
                ) : (
                  <EmptyState />
                )}
            </Bounds>
            <Environment preset="city" />
          </Suspense>
        </XR>
        <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={1.5} maxDistance={9} />
      </Canvas>
      {volume && (
        <div className="volume-stats">
          <span>{volume.id}</span>
          <span>{volume.mesh_vertices?.toLocaleString()} vertices</span>
          <span>{volume.kind === 'totalsegmentator' ? `${volume.anatomy_count || 0} anatomy meshes` : `${volume.source_slice_ids?.length || 0} source slice(s)`}</span>
          <span className={xrStatus.immersiveVR ? 'xr-ready' : 'xr-muted'}>
            {xrStatus.checked ? xrStatus.message : 'Checking spatial mode'}
          </span>
        </div>
      )}
    </section>
  );
}
